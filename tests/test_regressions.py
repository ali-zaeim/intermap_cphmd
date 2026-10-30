"""
Regression tests for logic bugs in the detection pipeline (water bridges,
interaction subsets, selections, rings, memory estimation and config).
"""
import copy
from collections import defaultdict

import numba
import numpy as np
import pytest
from bitarray import util as bu

import intermap.interactions.aro as aro_module
from intermap.interactions.waters import wb1
from intermap.managers.config import ConfigManager
from intermap.managers.indices import IndexManager
from intermap.runner import workflow


@pytest.fixture(autouse=True)
def fix_numba_recursion():
    """
    Overrides the conftest fixture: the pipeline needs the real Numba KDTree,
    as the SciPy replacement can't be passed to the jitted runpar.
    """
    yield


def run_t1(example_system_args, tmp_path, **kwargs):
    args = copy.deepcopy(example_system_args)
    args.output_dir = str(tmp_path)
    for key, value in kwargs.items():
        setattr(args, key, value)
    results = workflow(args)
    return {k: bu.sc_decode(v) for k, v in results.items()}


class TestWaterBridges:
    hb_cols = np.array([0, 1])

    def test_frames_stay_aligned_with_water_in_any_column(self):
        waters = np.array([100, 200], dtype=np.int32)
        resconv = np.arange(300, dtype=np.int32)
        ijf = np.array([[100, 1, 0], [2, 200, 0],
                        [100, 3, 1], [100, 4, 1],
                        [5, 200, 2], [6, 200, 2]], dtype=np.int32)
        inters = np.ones((6, 2), dtype=bool)

        res = wb1(ijf, inters, waters, self.hb_cols, resconv)
        assert sorted(map(tuple, res.tolist())) == [(3, 4, 100, 1),
                                                    (5, 6, 200, 2)]

    def test_single_hbond_column(self):
        waters = np.array([10], dtype=np.int32)
        resconv = np.arange(20, dtype=np.int32)
        ijf = np.array([[1, 10, 0], [2, 10, 0]], dtype=np.int32)
        inters = np.ones((2, 1), dtype=bool)

        res = wb1(ijf, inters, waters, np.array([0]), resconv)
        assert res.tolist() == [[1, 2, 10, 0]]

    def test_atomic_bridge_through_different_atoms_of_one_water(self):
        # Atom 1 donates to the water O (10); the water H (11) donates to 2
        waters = np.array([10, 11], dtype=np.int32)
        resconv = np.arange(20, dtype=np.int32)
        resconv[11] = 10
        ijf = np.array([[1, 10, 0], [11, 2, 0]], dtype=np.int32)
        inters = np.ones((2, 2), dtype=bool)

        res = wb1(ijf, inters, waters, self.hb_cols, resconv)
        assert res.tolist() == [[1, 2, 10, 0]]

    def test_atomic_no_bridge_within_one_residue(self):
        waters = np.array([10], dtype=np.int32)
        resconv = np.arange(20, dtype=np.int32)
        resconv[2] = 1
        ijf = np.array([[1, 10, 0], [2, 10, 0]], dtype=np.int32)
        inters = np.ones((2, 2), dtype=bool)

        res = wb1(ijf, inters, waters, self.hb_cols, resconv)
        assert res.shape == (0, 4)

    def test_residue_mode_has_no_self_bridges(self):
        # Two atoms of residue 1 and one of residue 2 bond water residue 100
        waters = np.array([10], dtype=np.int32)
        resconv = np.zeros(15, dtype=np.int32)
        resconv[10] = 100
        ijf = np.array([[1, 100, 0], [1, 100, 0], [2, 100, 0]],
                       dtype=np.int32)
        inters = np.ones((3, 2), dtype=bool)

        res = wb1(ijf, inters, waters, self.hb_cols, resconv, atomic=False)
        assert res.tolist() == [[1, 2, 100, 0]]


class TestPipeline:
    @pytest.mark.parametrize('inters', [
        ['CloseContact'],
        ['FaceToFace', 'EdgeToFace'],
        ['PiStacking'],
    ])
    def test_interaction_subsets(self, example_system_args, tmp_path,
                                 inters):
        results = run_t1(example_system_args, tmp_path,
                         interactions=np.asarray(inters))
        found = {k[-1] for k in results}
        assert found <= set(inters) | {'FaceToFace', 'EdgeToFace'}
        if inters == ['CloseContact']:
            assert found == {'CloseContact'}

    def test_no_frames_to_process(self, example_system_args, tmp_path):
        results = run_t1(example_system_args, tmp_path, start=1, last=1)
        assert results == {}

    def test_min_prevalence(self, example_system_args, tmp_path):
        results = run_t1(example_system_args, tmp_path, min_prevalence=100.0)
        assert len(results) > 0
        assert all(v.count(1) == len(v) for v in results.values())

    def test_estimate_uses_selected_coordinates(self, example_system_args,
                                                tmp_path, monkeypatch):
        n_atoms = []
        original = aro_module.get_aro_xyzs

        def spy(xyzs, *args):
            n_atoms.append(xyzs.shape[1])
            return original(xyzs, *args)

        monkeypatch.setattr(aro_module, 'get_aro_xyzs', spy)
        args = copy.deepcopy(example_system_args)
        args.output_dir = str(tmp_path)
        args.selection_1 = 'nucleic'
        args.selection_2 = 'resname LYS ARG'
        n_sel = IndexManager(args).sel_idx.size

        workflow(args)
        assert n_atoms and set(n_atoms) == {n_sel}


class TestSelections:
    def test_heavy_atom_selection_keeps_pairs_aligned(self,
                                                      example_system_args,
                                                      tmp_path):
        args = copy.deepcopy(example_system_args)
        args.output_dir = str(tmp_path)
        args.selection_1 = 'protein and not element H'
        iman = IndexManager(args)

        xyz = iman.universe.atoms.positions[iman.sel_idx]
        bonds = np.linalg.norm(xyz[iman.hb_don] - xyz[iman.hb_hydro], axis=1)
        assert iman.hb_don.size > 0
        assert bonds.max() < 1.3

    def test_fused_aromatic_rings(self, iman):
        sizes = defaultdict(list)
        for ring in iman.rings:
            first = iman.sel_idx[ring[0]]
            sizes[iman.universe.atoms[first].resindex].append(ring[-1])

        fused = iman.universe.select_atoms('resname TRP DA DG').residues
        assert len(fused) > 0
        for residue in fused:
            assert sorted(sizes[residue.resindex]) == [5, 6]


def test_n_procs_capped(tmp_path, get_clean_cfg, write_modified_cfg):
    original_threads = numba.get_num_threads()
    max_threads = numba.config.NUMBA_NUM_THREADS
    path = write_modified_cfg(tmp_path, get_clean_cfg, 'generals', 'n_procs',
                              str(max_threads + 1))
    try:
        ConfigManager(cfg_path=path)
        assert numba.get_num_threads() == max_threads
    finally:
        numba.set_num_threads(original_threads)
