# intermap/managers/cphmd.py
"""
Per-frame lambda-aware classification for GROMACS CpHMD trajectories.

Design constraints
------------------
- others() and aro() are @njit kernels receiving static index arrays.
  Per-frame state cannot be passed without recompilation overhead.
- Titratable atoms are therefore given every role they can take in any
  protonation state (upper bound), and roles that are not valid at a given
  frame are removed AFTER runpar() on (ijf_chunk, inters_chunk) using
  vectorised numpy -- no Python loops over pairs.

Column layout of inters_chunk (mirrors ContainerManager.get_inter_names):
  [aro_col_0, ..., aro_col_N, other_col_0, ..., other_col_M]
  where 'None' sentinels are EXCLUDED from both selected_aro and
  selected_others before indexing.

Lambda convention (GROMACS CpHMD, Aho et al. JCTC 2022):
  Single-site residues (ASPT, GLUT, LYST, ARGT):
    lambda < 0.5  -> protonated
    lambda >= 0.5 -> deprotonated

  HSPT (multisite, lambda_1 + lambda_2 + lambda_3 = 1):
    the state with the largest lambda is the current tautomer.
    Default order (HSP, HSE, HSD) follows the state charges of the CHARMM36
    CpHMD parameters: state 2 keeps HE2 (HSE), state 3 keeps HD1 (HSD).

Trajectory / lambda alignment:
  trajectory frame f is at time t0 + f * ps_per_frame, and the lambda value
  sampled closest to that time (from the .xvg time column) is used.

Roles controlled by lambda:
  - titratable protons exist only in protonated states (all interactions)
  - ionic atoms are charged only in their charged state
  - LYST NZ and HSPT ND1/NE2 accept H-bonds / coordinate metals only when
    they carry no proton
"""

import logging
import pathlib
import re

import numpy as np
import pandas as pd

logger = logging.getLogger('InterMapLogger')

CHARGE_CUTOFF = 0.5
DEFAULT_LAMBDA_GLOB = '**/*-coord-*.xvg'

# Single-site residues: atom roles that depend on the protonation state.
# Proton names are the hydrogens whose charge is 0 in the deprotonated state.
SINGLE_SITE = {
    'ASPT': {'anion': ['OD1', 'OD2'], 'proton': ['HD2']},
    'GLUT': {'anion': ['OE1', 'OE2'], 'proton': ['HE2']},
    'LYST': {'cation': ['NZ'], 'proton': ['HZ3'], 'acceptor': ['NZ']},
    'ARGT': {'cation': ['NE', 'NH1', 'NH2'], 'proton': ['HH12']},
}

HISTIDINE_RESNAME = 'HSPT'
HSPT_STATES = ('HSP', 'HSE', 'HSD')
HSPT_ROLES = {
    'HSP': {'cation': ['ND1', 'NE2'], 'proton': ['HD1', 'HE2']},
    'HSD': {'acceptor': ['NE2'], 'proton': ['HD1']},
    'HSE': {'acceptor': ['ND1'], 'proton': ['HE2']},
}

SUPPORTED_RESNAMES = frozenset(SINGLE_SITE) | {HISTIDINE_RESNAME}
ROLES = ('anion', 'cation', 'acceptor', 'proton')

# Interactions affected by each role, when the titratable atom is in the
# first (row 1) or second (row 2) column of ijf. 'proton' affects all.
ROLE_INTERACTIONS = {
    'anion': ({'Anionic', 'AnionPi'}, {'Cationic', 'PiAnion'}),
    'cation': ({'Cationic', 'CationPi'}, {'Anionic', 'PiCation'}),
    'acceptor': ({'HBAcceptor', 'XBAcceptor', 'MetalAcceptor'},
                 {'HBDonor', 'XBDonor', 'MetalDonor'}),
}


def register_protein_resnames():
    """
    Make MDAnalysis' 'protein' and 'backbone' selection keywords include the
    titratable residue names, which are otherwise silently left out.
    """
    from MDAnalysis.core.selection import ProteinSelection
    ProteinSelection.prot_res.update(SUPPORTED_RESNAMES)


# -- lambda file utilities ---------------------------------------------------

def parse_lambda_xvg(filepath):
    """
    Parse a GROMACS CpHMD .xvg file.
    Returns time : np.ndarray float64 (n,) in ps, lam : np.ndarray float32 (n,)
    """
    data = np.loadtxt(filepath, comments=('#', '@'), ndmin=2)
    if data.shape[0] == 0:
        raise ValueError(f"No data in {filepath}")
    return data[:, 0], data[:, 1].astype(np.float32)


def lambda_at_times(time, lam, frame_times, label=''):
    """
    Lambda value sampled closest to each trajectory frame time.

    Raises if the frames are not covered by the lambda time series.

    Returns frame_lam : np.ndarray float32 (n_frames,)
    """
    step = np.median(np.diff(time)) if time.size > 1 else 0.0
    if frame_times.size and (frame_times.min() < time[0] - step or
                             frame_times.max() > time[-1] + step):
        raise ValueError(
            f"CpHMD: lambda data {label} covers {time[0]}-{time[-1]} ps but "
            f"the analyzed frames span {frame_times.min()}-"
            f"{frame_times.max()} ps. Check lambda_ps_per_frame and that the "
            f"lambda files belong to the analyzed trajectory.")

    if time.size == 1:
        return np.full(frame_times.size, lam[0], dtype=np.float32)

    right = np.clip(np.searchsorted(time, frame_times), 1, time.size - 1)
    left = right - 1
    use_left = (frame_times - time[left]) <= (time[right] - frame_times)
    return lam[np.where(use_left, left, right)]


# -- reference file parser ---------------------------------------------------

def read_lambda_ref(path):
    """
    Parse the lambda reference file.
    Columns: resname  state  resid  [chain]  coord_file  (one header line)
    Returns pd.DataFrame with coord_num added (chain is None if absent).
    """
    rows = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) not in (4, 5):
            raise ValueError(
                f"Invalid line in lambda reference {path}: {line!r}. Expected"
                f" 'resname state resid [chain] coord_file'.")
        coord_file = parts[-1]
        m = re.search(r'-coord-(\d+)\.xvg', coord_file)
        rows.append(dict(
            resname=parts[0],
            state=int(parts[1]),
            resid=int(parts[2]),
            chain=parts[3] if len(parts) == 5 else None,
            coord_file=coord_file,
            coord_num=int(m.group(1)) if m else -1,
        ))
    return pd.DataFrame(rows)


# -- main class --------------------------------------------------------------

class CpHMDManager:
    """
    Reads per-residue lambda files and provides vectorised per-frame gating
    for InterMap's trajectory loop.

    Integration in runner.py
    ------------------------
    After IndexManager, before unpacking iman attributes:

        cphmd = CpHMDManager(
            lambda_ref_path = args.lambda_ref,
            lambda_dir      = args.lambda_dir,
            lambda_glob     = args.lambda_glob,
            traj_frames     = iman.traj_frames,
            ps_per_frame    = args.lambda_ps_per_frame,
            traj_t0         = universe.trajectory[0].time,
            traj_dt         = universe.trajectory.dt,
            hspt_states     = args.hspt_states,
        )
        cphmd.patch_index_manager(iman)

    After CutoffsManager (cuts), before estimate():

        gating_cols = CpHMDManager.get_gating_col_indices(
            cuts.selected_aro, cuts.selected_others)
        atom_lookup = cphmd.build_atom_lookup(iman)

    Inside the loop, immediately after runpar():

        if cphmd is not None and ijf_chunk.shape[0] > 0:
            ijf_chunk, inters_chunk = cphmd.gate_chunk(
                ijf_chunk, inters_chunk,
                contiguous[i], gating_cols, atom_lookup)
    """

    def __init__(self, lambda_ref_path, lambda_dir, lambda_glob,
                 traj_frames, ps_per_frame=None, traj_t0=0.0, traj_dt=None,
                 hspt_states=HSPT_STATES):
        self.traj_frames = np.asarray(traj_frames)
        self.n_frames = len(traj_frames)
        self.hspt_states = tuple(hspt_states)
        if sorted(self.hspt_states) != sorted(HSPT_ROLES):
            raise ValueError(
                f"CpHMD: hspt_states must order {sorted(HSPT_ROLES)}, got "
                f"{self.hspt_states}")
        self.ps_per_frame = self._get_ps_per_frame(ps_per_frame, traj_dt)
        self.frame_times = traj_t0 + self.traj_frames * self.ps_per_frame

        self.ref = read_lambda_ref(lambda_ref_path)
        self._xvg = self._find_xvg(lambda_dir, lambda_glob)
        self.sites = self._load_all()
        self.atom_roles = None
        self._log_summary()

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _get_ps_per_frame(ps_per_frame, traj_dt):
        if ps_per_frame is None:
            if not traj_dt:
                raise ValueError(
                    "CpHMD: the trajectory has no time step. Set "
                    "lambda_ps_per_frame in the [cphmd] section.")
            return float(traj_dt)

        if traj_dt and not np.isclose(ps_per_frame, traj_dt, rtol=1e-3):
            logger.warning(
                f"CpHMD: lambda_ps_per_frame={ps_per_frame} differs from the "
                f"trajectory time step ({traj_dt} ps). Using "
                f"lambda_ps_per_frame.")
        return float(ps_per_frame)

    @staticmethod
    def _find_xvg(direc, glob_pattern):
        result = {}
        for p in sorted(pathlib.Path(direc).glob(glob_pattern)):
            m = re.search(r'-coord-(\d+)\.xvg', str(p))
            if m:
                result.setdefault(int(m.group(1)), []).append(p)
        return result

    def _get_xvg(self, coord_num):
        paths = self._xvg.get(coord_num, [])
        if len(paths) > 1:
            raise ValueError(
                f"CpHMD: several .xvg files match coord_num={coord_num}: "
                f"{[str(x) for x in paths]}. Narrow lambda_dir or "
                f"lambda_glob so each coordinate file is found once.")
        return paths[0] if paths else None

    @staticmethod
    def _get_protonation_masks(frame_lam):
        is_protonated = frame_lam < CHARGE_CUTOFF
        return is_protonated, ~is_protonated

    def _get_hspt_tautomers(self, frame_lams):
        """Tautomer index (into self.hspt_states) with the largest lambda."""
        return np.argmax(np.vstack(frame_lams), axis=0)

    def _load_all(self):
        """
        One entry per titratable residue with its per-frame role masks:
        {'resname', 'resid', 'chain', 'roles': {role: {atom_name: mask}}}
        """
        skipped = sorted(set(self.ref['resname']) - SUPPORTED_RESNAMES)
        if skipped:
            logger.info(f"CpHMD: ignoring unsupported residues {skipped}")

        sites = []
        keys = ['resname', 'chain', 'resid']
        ref = self.ref[self.ref['resname'].isin(SUPPORTED_RESNAMES)]
        for (resname, chain, resid), rows in ref.groupby(keys, dropna=False,
                                                         sort=False):
            chain = None if pd.isna(chain) else chain
            label = f"{resname}{resid}" + (f":{chain}" if chain else '')
            rows = rows.sort_values('state')
            n_states = len(self.hspt_states) if (
                    resname == HISTIDINE_RESNAME) else 1
            if rows['state'].tolist() != list(range(1, n_states + 1)):
                raise ValueError(
                    f"CpHMD: {label} must have lambda states "
                    f"1..{n_states} in the reference, got "
                    f"{rows['state'].tolist()}.")

            frame_lams = []
            for _, row in rows.iterrows():
                xvg = self._get_xvg(int(row['coord_num']))
                if xvg is None:
                    break
                time, lam = parse_lambda_xvg(xvg)
                frame_lams.append(
                    lambda_at_times(time, lam, self.frame_times, label))
            if len(frame_lams) != n_states:
                logger.warning(
                    f"CpHMD: missing .xvg for {label} "
                    f"(coord files {rows['coord_file'].tolist()}). Skipping.")
                continue

            sites.append(dict(resname=resname, resid=int(resid), chain=chain,
                              label=label, frame_lams=frame_lams,
                              roles=self._get_roles(resname, frame_lams)))
        return sites

    def _get_roles(self, resname, frame_lams):
        """Per-frame activity of each (role, atom name) of a residue."""
        roles = {role: {} for role in ROLES}

        if resname == HISTIDINE_RESNAME:
            tautomer = self._get_hspt_tautomers(frame_lams)
            for i, state in enumerate(self.hspt_states):
                for role, names in HSPT_ROLES[state].items():
                    for name in names:
                        mask = roles[role].setdefault(
                            name, np.zeros(self.n_frames, dtype=bool))
                        mask |= tautomer == i
            return roles

        is_protonated, is_deprot = self._get_protonation_masks(frame_lams[0])
        active_when = {'anion': is_deprot, 'cation': is_protonated,
                       'acceptor': is_deprot, 'proton': is_protonated}
        for role, names in SINGLE_SITE[resname].items():
            for name in names:
                roles[role][name] = active_when[role]
        return roles

    def _log_summary(self):
        logger.info("CpHMD per-frame protonation summary:")
        for site in sorted(self.sites, key=lambda x: (str(x['chain']),
                                                       x['resid'])):
            if site['resname'] == HISTIDINE_RESNAME:
                tautomer = self._get_hspt_tautomers(site['frame_lams'])
                pops = ', '.join(
                    f"{state}={100.0 * np.mean(tautomer == i):.1f}%"
                    for i, state in enumerate(self.hspt_states))
                logger.info(f"  {site['label']}  {pops} of frames")
            else:
                is_protonated, _ = self._get_protonation_masks(
                    site['frame_lams'][0])
                logger.info(
                    f"  {site['label']}  protonated="
                    f"{100.0 * is_protonated.mean():.1f}% of frames  "
                    f"mean_lam={site['frame_lams'][0].mean():.3f}")

    @staticmethod
    def _select_residue(universe, site):
        """Atoms of a titratable residue; raises if it isn't unique."""
        query = f"resname {site['resname']} and resid {site['resid']}"
        if site['chain'] is not None:
            chain = site['chain']
            if hasattr(universe.atoms, 'chainIDs'):
                query += f" and (segid {chain} or chainID {chain})"
            else:
                query += f" and segid {chain}"

        atoms = universe.select_atoms(query)
        if len(atoms.residues) > 1:
            raise ValueError(
                f"CpHMD: '{query}' matches {len(atoms.residues)} residues "
                f"(segments {sorted(set(atoms.residues.segids))}). Add the "
                f"chain column to the lambda reference file.")
        return atoms

    @staticmethod
    def _to_sel_space(sel_idx, global_idx):
        import numpy_indexed as npi

        if len(global_idx) == 0:
            return np.array([], dtype=np.int32)
        raw = npi.indices(sel_idx,
                          np.unique(np.asarray(global_idx, dtype=np.int32)),
                          missing=-1)
        return raw[raw != -1].astype(np.int32)

    @staticmethod
    def _add(array, extra):
        if len(extra) == 0:
            return array
        return np.unique(np.concatenate([array, extra])).astype(np.int32)

    # -- public: static patch ------------------------------------------------

    def patch_index_manager(self, iman):
        """
        Give titratable atoms every role they can take in any state, as an
        upper bound so estimate() and ContainerManager pre-allocate enough
        space. Per-frame gating happens in gate_chunk().
        """
        universe = iman.universe
        sel_idx = iman.sel_idx

        self.atom_roles = []
        found = set()
        for site in self.sites:
            res_atoms = self._select_residue(universe, site)
            if not len(res_atoms):
                logger.warning(
                    f"CpHMD: {site['label']} not found in the topology. "
                    f"Skipping.")
                continue
            found.add(res_atoms.residues[0].resindex)

            for role, by_name in site['roles'].items():
                for name, active in by_name.items():
                    atom = res_atoms.select_atoms(f"name {name}")
                    if not len(atom):
                        logger.warning(
                            f"CpHMD: atom {name} not found in "
                            f"{site['label']}. Its {role} role is not gated.")
                        continue
                    sel_atom = self._to_sel_space(sel_idx, atom.indices)
                    if len(sel_atom):
                        self.atom_roles.append(dict(
                            role=role, sel_atom=int(sel_atom[0]),
                            active=active, label=f"{site['label']}:{name}"))

        added = {role: np.array(
            [x['sel_atom'] for x in self.atom_roles if x['role'] == role],
            dtype=np.int32) for role in ROLES}
        iman.anions = self._add(iman.anions, added['anion'])
        iman.cations = self._add(iman.cations, added['cation'])
        iman.hb_acc = self._add(iman.hb_acc, added['acceptor'])
        iman.met_acc = self._add(iman.met_acc, added['acceptor'])
        iman.xb_acc = self._add(iman.xb_acc, added['acceptor'])

        (iman.rings,
         iman.s1_cat, iman.s2_cat, iman.s1_ani, iman.s2_ani,
         iman.s1_cat_idx, iman.s2_cat_idx,
         iman.s1_ani_idx, iman.s2_ani_idx,
         iman.s1_rings, iman.s2_rings,
         iman.s1_rings_idx, iman.s2_rings_idx,
         iman.s1_aro_idx, iman.s2_aro_idx,
         iman.xyz_aro_idx) = iman.get_aro()

        # Titratable residues present in the topology but without lambdas
        titratable = universe.select_atoms(
            'resname ' + ' '.join(sorted(SUPPORTED_RESNAMES))).residues
        ungated = [f"{r.resname}{r.resid}" for r in titratable
                   if r.resindex not in found]
        if ungated:
            logger.warning(
                f"CpHMD: no lambda data for {ungated}. They keep the "
                f"protonation state of the topology.")

        logger.info(
            f"CpHMD static patch: anions={len(iman.anions)}, "
            f"cations={len(iman.cations)}, "
            f"hb_acceptors={len(iman.hb_acc)}, "
            f"gated atom roles={len(self.atom_roles)}")

    # -- public: interaction columns ----------------------------------------

    @staticmethod
    def _get_col_indices(selected_aro, selected_others, interaction_names):
        aro_names = [x for x in selected_aro if x != 'None']
        other_names = [x for x in selected_others if x != 'None']
        all_names = [*aro_names, *other_names]

        cols = [i for i, name in enumerate(all_names)
                if interaction_names is None or name in interaction_names]
        matched = [all_names[c] for c in cols]
        return np.array(cols, dtype=np.int32), matched

    @staticmethod
    def get_gating_col_indices(selected_aro, selected_others):
        """
        Columns to suppress per role, for the titratable atom in row 1 and
        in row 2 of ijf. A missing proton suppresses all the columns.
        """
        col_specs = dict(ROLE_INTERACTIONS, proton=(None, None))

        gating_cols = {}
        for role, (names_row1, names_row2) in col_specs.items():
            cols_row1, matched1 = CpHMDManager._get_col_indices(
                selected_aro, selected_others, names_row1)
            cols_row2, matched2 = CpHMDManager._get_col_indices(
                selected_aro, selected_others, names_row2)
            gating_cols[role] = (cols_row1, cols_row2)
            logger.info(
                f"CpHMD gating columns [{role}]: row1 {matched1}, "
                f"row2 {matched2}")
        return gating_cols

    # -- public: atom lookup -------------------------------------------------

    def build_atom_lookup(self, iman):
        """
        Build sel_idx-space lookup tables for CpHMD-aware gating.
        Call once after patch_index_manager(), before the trajectory loop.

        Returns
        -------
        dict role -> (atom_row, active)
            atom_row : np.ndarray int (n_sel,), row of each atom in `active`
                       or -1 if the atom has no gated role
            active   : np.ndarray bool (n_atoms_role, n_frames), whether the
                       atom can take the role at each analyzed frame
        """
        n_sel = len(iman.sel_idx)
        lookup = {}
        for role in ROLES:
            entries = [x for x in self.atom_roles if x['role'] == role]
            if not entries:
                continue
            atom_row = np.full(n_sel, -1, dtype=np.int64)
            atom_row[[x['sel_atom'] for x in entries]] = np.arange(
                len(entries))
            active = np.vstack([x['active'] for x in entries])
            lookup[role] = (atom_row, active)

        logger.info(
            "CpHMD atom lookup: " + ', '.join(
                f"{role}={lookup[role][1].shape[0]}" for role in lookup))
        return lookup

    # -- public: per-chunk gating -------------------------------------------

    @staticmethod
    def gate_chunk(ijf_chunk, inters_chunk, chunk_contiguous_indices,
                   gating_cols, atom_lookup):
        """
        Zero out interaction columns for pairs where the titratable residue was
        not allowed to express that role at that frame.

        Parameters
        ----------
        ijf_chunk               : np.ndarray int32 (n_pairs, 3)
                                  [atom_i, atom_j, local_frame_idx]
                                  atom indices are in sel_idx space.
        inters_chunk            : np.ndarray bool (n_pairs, n_inter_types)
        chunk_contiguous_indices: np.ndarray int (chunk_size,)
                                  local_frame_idx -> index in traj_frames
        gating_cols             : dict from get_gating_col_indices()
        atom_lookup             : dict from build_atom_lookup()

        Returns
        -------
        ijf_chunk, inters_chunk : with invalid CpHMD-dependent roles suppressed
                                  and all-False rows dropped.
        """
        if ijf_chunk.shape[0] == 0 or not atom_lookup:
            return ijf_chunk, inters_chunk

        frames = chunk_contiguous_indices[ijf_chunk[:, 2]]
        for role, (atom_row, active) in atom_lookup.items():
            for side, cols in enumerate(gating_cols[role]):
                if len(cols) == 0:
                    continue
                rows = atom_row[ijf_chunk[:, side]]
                gated = rows >= 0
                suppress = np.zeros(ijf_chunk.shape[0], dtype=bool)
                suppress[gated] = ~active[rows[gated], frames[gated]]
                if suppress.any():
                    inters_chunk[np.ix_(suppress, cols)] = False

        still_active = inters_chunk.any(axis=1)
        return ijf_chunk[still_active], inters_chunk[still_active]
