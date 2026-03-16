# intermap/managers/cphmd.py
"""
Per-frame lambda-aware classification for GROMACS CpHMD trajectories.

Design constraints
------------------
- others() and aro() are @njit kernels receiving static index arrays.
  Per-frame state cannot be passed without recompilation overhead.
- Gating happens AFTER runpar() on (ijf_chunk, inters_chunk) using
  vectorised numpy — no Python loops over pairs.

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
"""

import logging
import pathlib
import re
import numpy as np
import pandas as pd

logger = logging.getLogger('InterMapLogger')

PS_PER_FRAME      = 50
CHARGE_CUTOFF     = 0.5
HSPT_CATION_STATE = 1

ANION_ATOMS  = {'ASPT': ['OD1', 'OD2'], 'GLUT': ['OE1', 'OE2']}
CATION_ATOMS = {'HSPT': ['ND1', 'NE2']}

IONIC_INTERACTION_NAMES = frozenset({
    'Anionic', 'Cationic', 'AnionPi', 'PiAnion', 'PiCation', 'CationPi'
})


# ── lambda file utilities ─────────────────────────────────────────────────────

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
    n_lam  = len(lam_ps)
    if n_lam < needed:
        logger.warning(
            f"Lambda array ({n_lam} ps) shorter than needed ({needed} ps). "
            f"Padding with last value.")
        pad    = np.full(needed - n_lam, lam_ps[-1], dtype=np.float32)
        lam_ps = np.concatenate([lam_ps, pad])
    windows = lam_ps[:needed].reshape(n_traj_frames, ps_per_frame)
    return windows.mean(axis=1).astype(np.float32)


# ── reference file parser ─────────────────────────────────────────────────────

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
        resname    = parts[0]
        state      = int(parts[1])
        resid      = int(parts[2])
        coord_file = parts[3]
        m          = re.search(r'-coord-(\d+)\.xvg', coord_file)
        coord_num  = int(m.group(1)) if m else -1
        rows.append(dict(resname=resname, state=state, resid=resid,
                         coord_file=coord_file, coord_num=coord_num))
    return pd.DataFrame(rows)


# ── main class ────────────────────────────────────────────────────────────────

class CpHMDManager:
    """
    Reads per-residue lambda files and provides vectorised per-frame
    ionic-interaction gating for InterMap's trajectory loop.

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

        ionic_cols  = CpHMDManager.get_ionic_col_indices(
                          cuts.selected_aro, cuts.selected_others)
        atom_lookup = cphmd.build_atom_lookup(iman)

    Inside the loop, immediately after runpar():

        if cphmd is not None and ijf_chunk.shape[0] > 0:
            ijf_chunk, inters_chunk = cphmd.gate_chunk(
                ijf_chunk, inters_chunk,
                contiguous[i], ionic_cols, atom_lookup)
    """

    def __init__(self, lambda_ref_path, lambda_dir, lambda_glob,
                 traj_frames, ps_per_frame=PS_PER_FRAME):
        self.traj_frames  = traj_frames
        self.n_frames     = len(traj_frames)
        self.ps_per_frame = ps_per_frame

        self.ref         = read_lambda_ref(lambda_ref_path)
        self._xvg        = self._find_xvg(lambda_dir, lambda_glob)
        self.per_residue = self._load_all()
        self._log_summary()

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _find_xvg(direc, glob_pattern):
        result = {}
        for p in pathlib.Path(direc).glob(glob_pattern):
            m = re.search(r'-coord-(\d+)\.xvg', str(p))
            if m:
                result[int(m.group(1))] = p
        return result

    def _load_all(self):
        per_residue = {}
        for _, row in self.ref.iterrows():
            resname   = row['resname']
            resid     = int(row['resid'])
            state     = int(row['state'])
            coord_num = int(row['coord_num'])

            # ARGT/LYST always cationic — SMARTS catches them correctly
            if resname not in ('ASPT', 'GLUT', 'HSPT'):
                continue

            xvg = self._xvg.get(coord_num)
            if xvg is None:
                logger.warning(
                    f"CpHMD: no .xvg for coord_num={coord_num} "
                    f"({resname}{resid}). Skipping.")
                continue

            lam_ps    = parse_lambda_xvg(xvg)
            frame_lam = build_frame_lambda(lam_ps, self.n_frames,
                                           self.ps_per_frame)
            is_deprot = frame_lam >= CHARGE_CUTOFF   # bool (n_frames,)

            if resname in ('ASPT', 'GLUT'):
                is_charged = is_deprot        # anionic when deprotonated
                ionic_as   = 'anion'
            elif resname == 'HSPT':
                if state == HSPT_CATION_STATE:
                    is_charged = ~is_deprot   # cationic when protonated
                    ionic_as   = 'cation'
                else:
                    # Tautomer switching — never ionic
                    is_charged = np.zeros(self.n_frames, dtype=bool)
                    ionic_as   = 'neutral'

            per_residue[resid] = dict(
                resname    = resname,
                state      = state,
                coord_num  = coord_num,
                frame_lam  = frame_lam,
                is_charged = is_charged,
                ionic_as   = ionic_as,
            )
        return per_residue

    def _log_summary(self):
        logger.info("CpHMD per-frame protonation summary:")
        for resid, info in sorted(self.per_residue.items()):
            pct = 100.0 * info['is_charged'].mean()
            logger.info(
                f"  {info['resname']}{resid} state={info['state']} "
                f"ionic_as={info['ionic_as']}  "
                f"charged={pct:.1f}% of frames  "
                f"mean_λ={info['frame_lam'].mean():.3f}")

    # ── public: static patch ──────────────────────────────────────────────────

    def patch_index_manager(self, iman):
        """
        Add titratable atoms to iman.anions/cations as an upper-bound patch
        so estimate() and ContainerManager pre-allocate enough space.
        Per-frame gating happens in gate_chunk().
        """
        import numpy_indexed as npi

        universe  = iman.universe
        sel_idx   = iman.sel_idx
        anion_add, cation_add = [], []

        for resid, info in self.per_residue.items():
            ionic_as = info['ionic_as']
            if ionic_as == 'neutral':
                continue
            resname   = info['resname']
            res_sel   = universe.select_atoms(
                f"resname {resname} and resid {resid}")
            if not len(res_sel):
                continue
            atm_names = (ANION_ATOMS if ionic_as == 'anion'
                         else CATION_ATOMS).get(resname, [])
            charged   = res_sel.select_atoms(
                'name ' + ' '.join(atm_names)).indices
            (anion_add if ionic_as == 'anion' else cation_add).extend(charged)

        def to_sel(lst):
            if not lst:
                return np.array([], dtype=np.int32)
            raw = npi.indices(sel_idx,
                              np.unique(np.asarray(lst, dtype=np.int32)),
                              missing=-1)
            return raw[raw != -1].astype(np.int32)

        add_a = to_sel(anion_add)
        add_c = to_sel(cation_add)
        if len(add_a):
            iman.anions = np.unique(
                np.concatenate([iman.anions, add_a])).astype(np.int32)
        if len(add_c):
            iman.cations = np.unique(
                np.concatenate([iman.cations, add_c])).astype(np.int32)

        # Rebuild aromatic arrays (they depend on cations/anions)
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

    # ── public: ionic column indices ──────────────────────────────────────────

    @staticmethod
    def get_ionic_col_indices(selected_aro, selected_others):
        """
        Return column indices into inters_chunk for ionic interaction types.

        Column layout (matches ContainerManager.get_inter_names exactly):
          col 0 .. N-1 : selected_aro   entries that are not 'None'
          col N .. N+M-1: selected_others entries that are not 'None'

        selected_aro and selected_others are numba List objects from
        CutoffsManager — we filter out 'None' sentinels before indexing,
        mirroring what get_inter_names() does.

        Parameters
        ----------
        selected_aro    : numba List of str
        selected_others : numba List of str

        Returns
        -------
        np.ndarray int32  — column indices into inters_chunk
        """
        # Filter out 'None' sentinels exactly as get_inter_names() does
        aro_names   = [x for x in selected_aro    if x != 'None']
        other_names = [x for x in selected_others if x != 'None']

        cols  = []
        n_aro = len(aro_names)

        for i, name in enumerate(aro_names):
            if name in IONIC_INTERACTION_NAMES:
                cols.append(i)

        for i, name in enumerate(other_names):
            if name in IONIC_INTERACTION_NAMES:
                cols.append(n_aro + i)

        ionic_cols = np.array(cols, dtype=np.int32)

        matched = [([*aro_names, *other_names])[c] for c in cols]
        logger.info(
            f"CpHMD ionic column indices in inters_chunk: "
            f"{cols} -> {matched}")

        return ionic_cols

    # ── public: atom lookup (build once before loop) ──────────────────────────

    def build_atom_lookup(self, iman):
        """
        Build sel_idx-space atom index arrays for each titratable residue.
        Call once after patch_index_manager(), before the trajectory loop.

        Returns
        -------
        list of dicts:
            'sel_atoms'  : np.ndarray int32  (sel_idx space)
            'ionic_as'   : 'anion' | 'cation'
            'is_charged' : np.ndarray bool   (n_frames,)
        """
        import numpy_indexed as npi

        universe = iman.universe
        sel_idx  = iman.sel_idx
        lookup   = []

        for resid, info in self.per_residue.items():
            ionic_as = info['ionic_as']
            if ionic_as == 'neutral':
                continue
            resname = info['resname']
            res_sel = universe.select_atoms(
                f"resname {resname} and resid {resid}")
            if not len(res_sel):
                continue

            atm_names  = (ANION_ATOMS if ionic_as == 'anion'
                          else CATION_ATOMS).get(resname, [])
            global_idx = res_sel.select_atoms(
                'name ' + ' '.join(atm_names)).indices
            sel_space  = npi.indices(sel_idx,
                                     np.asarray(global_idx, dtype=np.int32),
                                     missing=-1)
            sel_space  = sel_space[sel_space != -1].astype(np.int32)
            if not len(sel_space):
                continue

            lookup.append(dict(
                sel_atoms  = sel_space,
                ionic_as   = ionic_as,
                is_charged = info['is_charged'],
            ))

        logger.info(
            f"CpHMD atom lookup: {len(lookup)} titratable sites with "
            f"ionic character")
        return lookup

    # ── public: per-chunk gating ──────────────────────────────────────────────

    def gate_chunk(self, ijf_chunk, inters_chunk,
                   chunk_contiguous_indices, ionic_cols, atom_lookup):
        """
        Zero out ionic interaction columns for pairs where the titratable
        residue was NOT in its charged state at that frame.

        Fully vectorised — the only loop is over titratable residues
        (typically 5-20), not over pairs.

        Parameters
        ----------
        ijf_chunk               : np.ndarray int32 (n_pairs, 3)
                                  [atom_i, atom_j, local_frame_idx]
                                  atom indices are in sel_idx space.
        inters_chunk            : np.ndarray bool (n_pairs, n_inter_types)
        chunk_contiguous_indices: np.ndarray int (chunk_size,)
                                  local_frame_idx -> contiguous global idx
                                  i.e. contiguous[i] from runner.py
        ionic_cols              : np.ndarray int32 from get_ionic_col_indices()
        atom_lookup             : list of dicts    from build_atom_lookup()

        Returns
        -------
        ijf_chunk, inters_chunk : with ionic cols suppressed where appropriate
                                  and all-False rows dropped.
        """
        if ijf_chunk.shape[0] == 0 or len(ionic_cols) == 0:
            return ijf_chunk, inters_chunk

        # Map each pair's local frame index to the contiguous global index
        # used to look up is_charged[].
        local_frames     = ijf_chunk[:, 2]
        local_clipped    = np.clip(local_frames, 0,
                                   len(chunk_contiguous_indices) - 1)
        global_frame_idx = chunk_contiguous_indices[local_clipped]  # (n_pairs,)

        pair_col0 = ijf_chunk[:, 0]   # atom_i in sel_idx space
        pair_col1 = ijf_chunk[:, 1]   # atom_j

        for entry in atom_lookup:
            sel_atoms  = entry['sel_atoms']    # int32 (n_charged_atoms,)
            is_charged = entry['is_charged']   # bool  (n_frames,)

            # Which pairs involve one of this residue's charged atoms?
            pair_involves = (
                np.isin(pair_col0, sel_atoms) |
                np.isin(pair_col1, sel_atoms)
            )                                              # bool (n_pairs,)

            if not pair_involves.any():
                continue

            # Is the residue in its charged state at each pair's frame?
            charged_at_frame = is_charged[global_frame_idx]   # bool (n_pairs,)

            # Suppress where: pair involves this residue AND uncharged that frame
            suppress = pair_involves & ~charged_at_frame       # bool (n_pairs,)

            if suppress.any():
                inters_chunk[np.ix_(suppress, ionic_cols)] = False

        # Drop pairs that now have no remaining interactions
        still_active = inters_chunk.any(axis=1)
        return ijf_chunk[still_active], inters_chunk[still_active]