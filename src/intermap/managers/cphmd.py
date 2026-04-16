# intermap/managers/cphmd.py
"""
Per-frame lambda-aware classification for GROMACS CpHMD trajectories.

Design constraints
------------------
- others() and aro() are @njit kernels receiving static index arrays.
  Per-frame state cannot be passed without recompilation overhead.
- Gating happens AFTER runpar() on (ijf_chunk, inters_chunk) using
  vectorised numpy -- no Python loops over pairs.

Column layout of inters_chunk (mirrors ContainerManager.get_inter_names):
  [aro_col_0, ..., aro_col_N, other_col_0, ..., other_col_M]
  where 'None' sentinels are EXCLUDED from both selected_aro and
  selected_others before indexing.

Lambda convention (GROMACS CpHMD, this force field):
  lambda < 0.5  ->  protonated
  lambda >= 0.5 ->  deprotonated

  ASPT / GLUT:  deprot -> anionic
  HSPT state 1: prot   -> cationic  (doubly protonated)
  HSPT state 2/3: tautomer only, never ionic

Trajectory / lambda alignment:
  frame f  <->  lambda window [f*50 : (f+1)*50] ps, mean of window used.

CpHMD-specific sidechain chemistry gating:
  - acidic titratable hydrogens are treated as lambda-controlled visibility
  - charged HSPT ND1/NE2 cannot act as acceptors while protonated
"""

import logging
import pathlib
import re

import numpy as np
import pandas as pd

logger = logging.getLogger('InterMapLogger')

PS_PER_FRAME = 50
CHARGE_CUTOFF = 0.5
HSPT_CATION_STATE = 1

ACIDIC_RESNAMES = frozenset({'ASPT', 'GLUT'})
HISTIDINE_RESNAME = 'HSPT'
SUPPORTED_RESNAMES = ACIDIC_RESNAMES | {HISTIDINE_RESNAME}

ANION_ATOMS = {'ASPT': ['OD1', 'OD2'], 'GLUT': ['OE1', 'OE2']}
CATION_ATOMS = {'HSPT': ['ND1', 'NE2']}

IONIC_INTERACTION_NAMES = frozenset({
    'Anionic', 'Cationic', 'AnionPi', 'PiAnion', 'PiCation', 'CationPi'
})
ACCEPTOR_ROW1_INTERACTION_NAMES = frozenset({
    'HBAcceptor', 'XBAcceptor', 'MetalAcceptor'
})
ACCEPTOR_ROW2_INTERACTION_NAMES = frozenset({
    'HBDonor', 'XBDonor', 'MetalDonor'
})
DONOR_H_ROW1_INTERACTION_NAMES = frozenset({'HBDonor'})
DONOR_H_ROW2_INTERACTION_NAMES = frozenset({'HBAcceptor'})


# -- lambda file utilities ---------------------------------------------------

def parse_lambda_xvg(filepath):
    """
    Parse a GROMACS CpHMD .xvg file.
    Returns lam : np.ndarray float32 (n_ps,)
    """
    data_lines = []
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith(('#', '@')):
                data_lines.append(s)
    if not data_lines:
        raise ValueError(f"No data in {filepath}")
    data = np.loadtxt(data_lines)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 1].astype(np.float32)


def build_frame_lambda(lam_ps, n_traj_frames, ps_per_frame=PS_PER_FRAME):
    """
    Average per-ps lambda into per-trajectory-frame lambda via reshape+mean.
    No Python loop over frames.

    Returns frame_lam : np.ndarray float32 (n_traj_frames,)
    """
    needed = n_traj_frames * ps_per_frame
    n_lam = len(lam_ps)
    if n_lam < needed:
        logger.warning(
            f"Lambda array ({n_lam} ps) shorter than needed ({needed} ps). "
            f"Padding with last value.")
        pad = np.full(needed - n_lam, lam_ps[-1], dtype=np.float32)
        lam_ps = np.concatenate([lam_ps, pad])
    windows = lam_ps[:needed].reshape(n_traj_frames, ps_per_frame)
    return windows.mean(axis=1).astype(np.float32)


# -- reference file parser ---------------------------------------------------

def read_lambda_ref(path):
    """
    Parse the lambda reference file.
    Columns: resname  state  resid  coord_file  (one header line)
    Returns pd.DataFrame with coord_num added.
    """
    rows = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split()
        if not parts:
            continue
        resname = parts[0]
        state = int(parts[1])
        resid = int(parts[2])
        coord_file = parts[3]
        m = re.search(r'-coord-(\d+)\.xvg', coord_file)
        coord_num = int(m.group(1)) if m else -1
        rows.append(dict(
            resname=resname,
            state=state,
            resid=resid,
            coord_file=coord_file,
            coord_num=coord_num,
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
            lambda_glob     = getattr(args, 'lambda_glob',
                                      '**/eq/*-coord-*.xvg'),
            traj_frames     = iman.traj_frames,
            ps_per_frame    = getattr(args, 'lambda_ps_per_frame', 50),
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
                 traj_frames, ps_per_frame=PS_PER_FRAME):
        self.traj_frames = traj_frames
        self.n_frames = len(traj_frames)
        self.ps_per_frame = ps_per_frame

        self.ref = read_lambda_ref(lambda_ref_path)
        self._xvg = self._find_xvg(lambda_dir, lambda_glob)
        self.site_entries = self._load_all()
        self.sites_by_resid = self._group_sites_by_resid()
        self.per_residue = self.sites_by_resid
        self._log_summary()

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _find_xvg(direc, glob_pattern):
        result = {}
        for p in pathlib.Path(direc).glob(glob_pattern):
            m = re.search(r'-coord-(\d+)\.xvg', str(p))
            if not m:
                continue
            coord_num = int(m.group(1))
            prev = result.get(coord_num)
            if prev is not None and prev != p:
                logger.warning(
                    f"CpHMD: duplicate .xvg matches for coord_num={coord_num}: "
                    f"{prev} and {p}. Using {p}.")
            result[coord_num] = p
        return result

    def _group_sites_by_resid(self):
        grouped = {}
        for entry in self.site_entries:
            grouped.setdefault(entry['resid'], []).append(entry)
        return grouped

    def _load_all(self):
        site_entries = []
        for _, row in self.ref.iterrows():
            resname = row['resname']
            resid = int(row['resid'])
            state = int(row['state'])
            coord_num = int(row['coord_num'])

            if resname not in SUPPORTED_RESNAMES:
                continue

            xvg = self._xvg.get(coord_num)
            if xvg is None:
                logger.warning(
                    f"CpHMD: no .xvg for coord_num={coord_num} "
                    f"({resname}{resid} state={state}). Skipping.")
                continue

            lam_ps = parse_lambda_xvg(xvg)
            frame_lam = build_frame_lambda(lam_ps, self.n_frames,
                                           self.ps_per_frame)
            is_deprot = frame_lam >= CHARGE_CUTOFF
            is_protonated = ~is_deprot

            if resname in ACIDIC_RESNAMES:
                is_charged = is_deprot
                ionic_as = 'anion'
            elif resname == HISTIDINE_RESNAME:
                if state == HSPT_CATION_STATE:
                    is_charged = is_protonated
                    ionic_as = 'cation'
                else:
                    is_charged = np.zeros(self.n_frames, dtype=bool)
                    ionic_as = 'neutral'
            else:
                continue

            site_entries.append(dict(
                resid=resid,
                resname=resname,
                state=state,
                coord_num=coord_num,
                frame_lam=frame_lam,
                is_protonated=is_protonated,
                is_charged=is_charged,
                ionic_as=ionic_as,
            ))
        return site_entries

    def _log_summary(self):
        logger.info("CpHMD per-frame protonation summary:")
        for info in sorted(self.site_entries,
                           key=lambda x: (x['resid'], x['state'])):
            pct = 100.0 * info['is_charged'].mean()
            logger.info(
                f"  {info['resname']}{info['resid']} state={info['state']} "
                f"ionic_as={info['ionic_as']}  "
                f"charged={pct:.1f}% of frames  "
                f"mean_lam={info['frame_lam'].mean():.3f}")

    @staticmethod
    def _or_masks(entries, field, n_frames):
        mask = np.zeros(n_frames, dtype=bool)
        for entry in entries:
            mask |= entry[field]
        return mask

    @staticmethod
    def _get_col_indices(selected_aro, selected_others, interaction_names):
        aro_names = [x for x in selected_aro if x != 'None']
        other_names = [x for x in selected_others if x != 'None']

        cols = []
        n_aro = len(aro_names)

        for i, name in enumerate(aro_names):
            if name in interaction_names:
                cols.append(i)

        for i, name in enumerate(other_names):
            if name in interaction_names:
                cols.append(n_aro + i)

        all_names = [*aro_names, *other_names]
        matched = [all_names[c] for c in cols]
        return np.array(cols, dtype=np.int32), matched

    @staticmethod
    def _bonded_hydrogens(atom_group):
        hydrogens = []
        for atom in atom_group:
            for bonded in atom.bonded_atoms:
                name = getattr(bonded, 'name', '')
                element = getattr(bonded, 'element', '')
                if element == 'H' or name.startswith('H'):
                    hydrogens.append(bonded.index)
        if not hydrogens:
            return np.array([], dtype=np.int32)
        return np.unique(np.asarray(hydrogens, dtype=np.int32))

    @staticmethod
    def _to_sel_space(sel_idx, global_idx):
        import numpy_indexed as npi

        if len(global_idx) == 0:
            return np.array([], dtype=np.int32)
        raw = npi.indices(sel_idx,
                          np.unique(np.asarray(global_idx, dtype=np.int32)),
                          missing=-1)
        return raw[raw != -1].astype(np.int32)

    # -- public: static patch ------------------------------------------------

    def patch_index_manager(self, iman):
        """
        Add titratable atoms to iman.anions/cations as an upper-bound patch
        so estimate() and ContainerManager pre-allocate enough space.
        Per-frame gating happens in gate_chunk().
        """
        universe = iman.universe
        sel_idx = iman.sel_idx
        anion_add, cation_add = [], []

        for resid, entries in self.sites_by_resid.items():
            ionic_entries = [x for x in entries if x['ionic_as'] != 'neutral']
            if not ionic_entries:
                continue

            ionic_as = ionic_entries[0]['ionic_as']
            resname = ionic_entries[0]['resname']
            res_sel = universe.select_atoms(
                f"resname {resname} and resid {resid}")
            if not len(res_sel):
                continue

            atm_names = (ANION_ATOMS if ionic_as == 'anion'
                         else CATION_ATOMS).get(resname, [])
            charged = res_sel.select_atoms(
                'name ' + ' '.join(atm_names)).indices
            if ionic_as == 'anion':
                anion_add.extend(charged)
            else:
                cation_add.extend(charged)

        add_a = self._to_sel_space(sel_idx, anion_add)
        add_c = self._to_sel_space(sel_idx, cation_add)
        if len(add_a):
            iman.anions = np.unique(
                np.concatenate([iman.anions, add_a])).astype(np.int32)
        if len(add_c):
            iman.cations = np.unique(
                np.concatenate([iman.cations, add_c])).astype(np.int32)

        (iman.rings,
         iman.s1_cat, iman.s2_cat, iman.s1_ani, iman.s2_ani,
         iman.s1_cat_idx, iman.s2_cat_idx,
         iman.s1_ani_idx, iman.s2_ani_idx,
         iman.s1_rings, iman.s2_rings,
         iman.s1_rings_idx, iman.s2_rings_idx,
         iman.s1_aro_idx, iman.s2_aro_idx,
         iman.xyz_aro_idx) = iman.get_aro()

        logger.info(
            f"CpHMD static patch: anions={len(iman.anions)}, "
            f"cations={len(iman.cations)}")

    # -- public: interaction columns ----------------------------------------

    @staticmethod
    def get_ionic_col_indices(selected_aro, selected_others):
        ionic_cols, matched = CpHMDManager._get_col_indices(
            selected_aro, selected_others, IONIC_INTERACTION_NAMES)
        logger.info(
            f"CpHMD ionic column indices in inters_chunk: "
            f"{ionic_cols.tolist()} -> {matched}")
        return ionic_cols

    @staticmethod
    def get_gating_col_indices(selected_aro, selected_others):
        col_specs = {
            'ionic': IONIC_INTERACTION_NAMES,
            'acceptor_row1': ACCEPTOR_ROW1_INTERACTION_NAMES,
            'acceptor_row2': ACCEPTOR_ROW2_INTERACTION_NAMES,
            'donor_h_row1': DONOR_H_ROW1_INTERACTION_NAMES,
            'donor_h_row2': DONOR_H_ROW2_INTERACTION_NAMES,
        }

        gating_cols = {}
        for key, names in col_specs.items():
            cols, matched = CpHMDManager._get_col_indices(
                selected_aro, selected_others, names)
            gating_cols[key] = cols
            logger.info(
                f"CpHMD gating columns [{key}]: "
                f"{cols.tolist()} -> {matched}")
        return gating_cols

    # -- public: atom lookup -------------------------------------------------

    def build_atom_lookup(self, iman):
        """
        Build sel_idx-space atom index arrays for CpHMD-aware gating.
        Call once after patch_index_manager(), before the trajectory loop.

        Returns
        -------
        dict with three lists:
            'ionic'    : atoms whose ionic interactions are lambda-gated
            'donor_h'  : titratable hydrogens whose H-bond donor role is gated
            'acceptor' : titratable atoms whose acceptor role is gated
        """
        universe = iman.universe
        sel_idx = iman.sel_idx
        lookup = {'ionic': [], 'donor_h': [], 'acceptor': []}

        for resid, entries in self.sites_by_resid.items():
            resname = entries[0]['resname']
            res_sel = universe.select_atoms(
                f"resname {resname} and resid {resid}")
            if not len(res_sel):
                continue

            ionic_entries = [x for x in entries if x['ionic_as'] != 'neutral']
            if ionic_entries:
                ionic_as = ionic_entries[0]['ionic_as']
                atom_names = (ANION_ATOMS if ionic_as == 'anion'
                              else CATION_ATOMS).get(resname, [])
                global_idx = res_sel.select_atoms(
                    'name ' + ' '.join(atom_names)).indices
                sel_space = self._to_sel_space(sel_idx, global_idx)
                if len(sel_space):
                    lookup['ionic'].append(dict(
                        resid=resid,
                        resname=resname,
                        sel_atoms=sel_space,
                        is_charged=self._or_masks(
                            ionic_entries, 'is_charged', self.n_frames),
                    ))

            if resname in ACIDIC_RESNAMES:
                titratable_atoms = res_sel.select_atoms(
                    'name ' + ' '.join(ANION_ATOMS[resname]))
                donor_h_global = self._bonded_hydrogens(titratable_atoms)
                donor_h_sel = self._to_sel_space(sel_idx, donor_h_global)
                if len(donor_h_sel):
                    lookup['donor_h'].append(dict(
                        resid=resid,
                        resname=resname,
                        sel_atoms=donor_h_sel,
                        is_visible=self._or_masks(
                            entries, 'is_protonated', self.n_frames),
                    ))

            if resname == HISTIDINE_RESNAME:
                cation_entries = [
                    x for x in entries if x['state'] == HSPT_CATION_STATE
                ]
                if cation_entries:
                    acceptor_atoms = res_sel.select_atoms(
                        'name ' + ' '.join(CATION_ATOMS[resname])).indices
                    acceptor_sel = self._to_sel_space(sel_idx, acceptor_atoms)
                    if len(acceptor_sel):
                        is_cationic = self._or_masks(
                            cation_entries, 'is_protonated', self.n_frames)
                        lookup['acceptor'].append(dict(
                            resid=resid,
                            resname=resname,
                            sel_atoms=acceptor_sel,
                            is_allowed=~is_cationic,
                        ))

        logger.info(
            f"CpHMD atom lookup: ionic={len(lookup['ionic'])}, "
            f"donor_h={len(lookup['donor_h'])}, "
            f"acceptor={len(lookup['acceptor'])}")
        return lookup

    # -- public: per-chunk gating -------------------------------------------

    def gate_chunk(self, ijf_chunk, inters_chunk,
                   chunk_contiguous_indices, gating_cols, atom_lookup):
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
                                  local_frame_idx -> contiguous global idx
        gating_cols             : dict of np.ndarray int32 column indices
        atom_lookup             : dict from build_atom_lookup()

        Returns
        -------
        ijf_chunk, inters_chunk : with invalid CpHMD-dependent roles suppressed
                                  and all-False rows dropped.
        """
        if ijf_chunk.shape[0] == 0:
            return ijf_chunk, inters_chunk

        if atom_lookup is None:
            return ijf_chunk, inters_chunk

        has_lookup = any(len(atom_lookup[key]) > 0 for key in atom_lookup)
        has_cols = any(len(gating_cols[key]) > 0 for key in gating_cols)
        if not has_lookup or not has_cols:
            return ijf_chunk, inters_chunk

        local_frames = ijf_chunk[:, 2]
        local_clipped = np.clip(local_frames, 0,
                                len(chunk_contiguous_indices) - 1)
        global_frame_idx = chunk_contiguous_indices[local_clipped]

        pair_col0 = ijf_chunk[:, 0]
        pair_col1 = ijf_chunk[:, 1]

        ionic_cols = gating_cols['ionic']
        for entry in atom_lookup['ionic']:
            if len(ionic_cols) == 0:
                break
            sel_atoms = entry['sel_atoms']
            pair_involves = (
                np.isin(pair_col0, sel_atoms) |
                np.isin(pair_col1, sel_atoms)
            )
            if not pair_involves.any():
                continue
            charged_at_frame = entry['is_charged'][global_frame_idx]
            suppress = pair_involves & ~charged_at_frame
            if suppress.any():
                inters_chunk[np.ix_(suppress, ionic_cols)] = False

        donor_h_row1 = gating_cols['donor_h_row1']
        donor_h_row2 = gating_cols['donor_h_row2']
        for entry in atom_lookup['donor_h']:
            sel_atoms = entry['sel_atoms']
            visible_at_frame = entry['is_visible'][global_frame_idx]

            if len(donor_h_row1):
                donor_row1 = np.isin(pair_col0, sel_atoms)
                suppress = donor_row1 & ~visible_at_frame
                if suppress.any():
                    inters_chunk[np.ix_(suppress, donor_h_row1)] = False

            if len(donor_h_row2):
                donor_row2 = np.isin(pair_col1, sel_atoms)
                suppress = donor_row2 & ~visible_at_frame
                if suppress.any():
                    inters_chunk[np.ix_(suppress, donor_h_row2)] = False

        acceptor_row1 = gating_cols['acceptor_row1']
        acceptor_row2 = gating_cols['acceptor_row2']
        for entry in atom_lookup['acceptor']:
            sel_atoms = entry['sel_atoms']
            allowed_at_frame = entry['is_allowed'][global_frame_idx]

            if len(acceptor_row1):
                acceptor_in_row1 = np.isin(pair_col0, sel_atoms)
                suppress = acceptor_in_row1 & ~allowed_at_frame
                if suppress.any():
                    inters_chunk[np.ix_(suppress, acceptor_row1)] = False

            if len(acceptor_row2):
                acceptor_in_row2 = np.isin(pair_col1, sel_atoms)
                suppress = acceptor_in_row2 & ~allowed_at_frame
                if suppress.any():
                    inters_chunk[np.ix_(suppress, acceptor_row2)] = False

        still_active = inters_chunk.any(axis=1)
        return ijf_chunk[still_active], inters_chunk[still_active]
