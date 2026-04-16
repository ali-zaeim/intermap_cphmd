import numpy as np

from intermap.managers.cphmd import CpHMDManager


def make_manager():
    return CpHMDManager.__new__(CpHMDManager)


def empty_cols():
    return np.array([], dtype=np.int32)


def test_get_protonation_masks_keeps_default_cutoff_for_acidic_sites():
    frame_lam = np.array([0.2, 0.5, 0.8], dtype=np.float32)

    is_protonated, is_deprotonated = CpHMDManager._get_protonation_masks(
        'ASPT', 1, frame_lam)

    assert is_protonated.tolist() == [True, False, False]
    assert is_deprotonated.tolist() == [False, True, True]


def test_get_protonation_masks_inverts_cutoff_for_histidine_state_1():
    frame_lam = np.array([0.2, 0.5, 0.8], dtype=np.float32)

    is_protonated, is_deprotonated = CpHMDManager._get_protonation_masks(
        'HSPT', 1, frame_lam)

    assert is_protonated.tolist() == [False, True, True]
    assert is_deprotonated.tolist() == [True, False, False]


def test_gate_chunk_suppresses_ionic_only():
    manager = make_manager()

    ijf_chunk = np.array([[1, 5, 0]], dtype=np.int32)
    inters_chunk = np.array([[True, True]], dtype=bool)
    gating_cols = {
        'ionic': np.array([0], dtype=np.int32),
        'acceptor_row1': empty_cols(),
        'acceptor_row2': empty_cols(),
        'donor_h_row1': empty_cols(),
        'donor_h_row2': empty_cols(),
    }
    atom_lookup = {
        'ionic': [{'sel_atoms': np.array([1], dtype=np.int32),
                   'is_charged': np.array([False], dtype=bool)}],
        'donor_h': [],
        'acceptor': [],
    }

    gated_ijf, gated_inters = manager.gate_chunk(
        ijf_chunk, inters_chunk, np.array([0], dtype=np.int32),
        gating_cols, atom_lookup)

    assert gated_ijf.shape == (1, 3)
    assert gated_inters.tolist() == [[False, True]]


def test_gate_chunk_suppresses_hidden_donor_hydrogen_on_both_sides():
    manager = make_manager()

    ijf_chunk = np.array([[7, 10, 0], [10, 7, 0]], dtype=np.int32)
    inters_chunk = np.array([[True, False], [False, True]], dtype=bool)
    gating_cols = {
        'ionic': empty_cols(),
        'acceptor_row1': empty_cols(),
        'acceptor_row2': empty_cols(),
        'donor_h_row1': np.array([0], dtype=np.int32),
        'donor_h_row2': np.array([1], dtype=np.int32),
    }
    atom_lookup = {
        'ionic': [],
        'donor_h': [{'sel_atoms': np.array([7], dtype=np.int32),
                     'is_visible': np.array([False], dtype=bool)}],
        'acceptor': [],
    }

    gated_ijf, gated_inters = manager.gate_chunk(
        ijf_chunk, inters_chunk, np.array([0], dtype=np.int32),
        gating_cols, atom_lookup)

    assert gated_ijf.shape == (0, 3)
    assert gated_inters.shape == (0, 2)


def test_gate_chunk_suppresses_acceptor_roles_on_both_sides():
    manager = make_manager()

    ijf_chunk = np.array([[9, 4, 0], [4, 9, 0]], dtype=np.int32)
    inters_chunk = np.array([
        [True, True, False, False],
        [False, False, True, True],
    ], dtype=bool)
    gating_cols = {
        'ionic': empty_cols(),
        'acceptor_row1': np.array([0, 1], dtype=np.int32),
        'acceptor_row2': np.array([2, 3], dtype=np.int32),
        'donor_h_row1': empty_cols(),
        'donor_h_row2': empty_cols(),
    }
    atom_lookup = {
        'ionic': [],
        'donor_h': [],
        'acceptor': [{'sel_atoms': np.array([9], dtype=np.int32),
                      'is_allowed': np.array([False], dtype=bool)}],
    }

    gated_ijf, gated_inters = manager.gate_chunk(
        ijf_chunk, inters_chunk, np.array([0], dtype=np.int32),
        gating_cols, atom_lookup)

    assert gated_ijf.shape == (0, 3)
    assert gated_inters.shape == (0, 4)
