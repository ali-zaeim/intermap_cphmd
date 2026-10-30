import logging
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pytest

from intermap.managers.config import ConfigManager
from intermap.managers.cphmd import (CpHMDManager, lambda_at_times,
                                     read_lambda_ref)


def make_manager():
    return CpHMDManager.__new__(CpHMDManager)


def empty_cols():
    return np.array([], dtype=np.int32)


def cols(*values):
    return np.array(values, dtype=np.int32)


def gating(**roles):
    """gating_cols with (row1, row2) columns for the given roles."""
    out = {role: (empty_cols(), empty_cols())
           for role in ('anion', 'cation', 'acceptor', 'proton')}
    out.update(roles)
    return out


def lookup(role, atom, active, n_sel=20):
    """atom_lookup with a single gated atom."""
    atom_row = np.full(n_sel, -1, dtype=np.int64)
    atom_row[atom] = 0
    return {role: (atom_row, np.array([active], dtype=bool))}


def write_xvg(path, time, lam):
    with open(path, 'w') as f:
        f.write('# This file was created by gmx cphmd\n')
        f.write('@    xaxis  label "Time (ps)"\n')
        f.write('@TYPE xy\n')
        for t, x in zip(time, lam):
            f.write(f'{t:10.4f} {x:10.4f}\n')


# -- protonation states -------------------------------------------------------

def test_get_protonation_masks_uses_half_cutoff():
    frame_lam = np.array([0.2, 0.5, 0.8], dtype=np.float32)

    is_protonated, is_deprotonated = CpHMDManager._get_protonation_masks(
        frame_lam)

    assert is_protonated.tolist() == [True, False, False]
    assert is_deprotonated.tolist() == [False, True, True]


def test_hspt_roles_follow_the_dominant_tautomer():
    manager = make_manager()
    manager.hspt_states = ('HSP', 'HSE', 'HSD')
    manager.n_frames = 3
    # frame 0: HSP, frame 1: HSE, frame 2: HSD
    lams = [np.array([0.9, 0.1, 0.0]), np.array([0.1, 0.8, 0.1]),
            np.array([0.0, 0.1, 0.9])]

    roles = manager._get_roles('HSPT', lams)

    assert roles['cation']['ND1'].tolist() == [True, False, False]
    assert roles['cation']['NE2'].tolist() == [True, False, False]
    assert roles['acceptor']['ND1'].tolist() == [False, True, False]
    assert roles['acceptor']['NE2'].tolist() == [False, False, True]
    assert roles['proton']['HD1'].tolist() == [True, False, True]
    assert roles['proton']['HE2'].tolist() == [True, True, False]


def test_lyst_roles():
    manager = make_manager()
    manager.n_frames = 2

    roles = manager._get_roles('LYST', [np.array([0.1, 0.9])])

    assert roles['cation']['NZ'].tolist() == [True, False]
    assert roles['acceptor']['NZ'].tolist() == [False, True]
    assert roles['proton']['HZ3'].tolist() == [True, False]


# -- lambda files ------------------------------------------------------------

def test_lambda_at_times_uses_time_column_and_frame_times():
    time = np.arange(0, 100, 2.0)              # written every 2 ps
    lam = (time / 100).astype(np.float32)
    frame_times = 0.0 + np.array([5, 7]) * 10  # start=5, stride=2, 10 ps

    frame_lam = lambda_at_times(time, lam, frame_times)

    assert np.allclose(frame_lam, [0.5, 0.7])


def test_lambda_at_times_rejects_uncovered_frames():
    time = np.arange(0, 100, 1.0)
    lam = np.zeros(time.size, dtype=np.float32)

    with pytest.raises(ValueError, match='covers'):
        lambda_at_times(time, lam, np.array([0.0, 500.0]))


def test_read_lambda_ref_with_and_without_chain(tmp_path):
    ref = tmp_path / 'lambdareference.dat'
    ref.write_text('resname state# resid chain coordinateFile\n'
                   'GLUT 1   13   cphmd-coord-1.xvg\n'
                   'HSPT 2   75 B cphmd-coord-4.xvg\n')

    df = read_lambda_ref(ref)

    assert df['chain'].tolist() == [None, 'B']
    assert df['coord_num'].tolist() == [1, 4]
    assert df['resid'].tolist() == [13, 75]


# -- full manager on a toy topology --------------------------------------------

RESIDUES = [
    ('ASPT', 52, 'PROA', ['CG', 'OD1', 'OD2', 'HD2']),
    ('HSPT', 75, 'PROA', ['ND1', 'HD1', 'NE2', 'HE2']),
    ('LYST', 122, 'PROA', ['NZ', 'HZ1', 'HZ3']),
    ('ASPT', 52, 'PROB', ['CG', 'OD1', 'OD2', 'HD2']),
]


def make_iman():
    names, resindex = [], []
    for i, (_, _, _, atoms) in enumerate(RESIDUES):
        names += atoms
        resindex += [i] * len(atoms)
    segids = ['PROA', 'PROB']
    u = mda.Universe.empty(len(names), n_residues=len(RESIDUES),
                           n_segments=len(segids), atom_resindex=resindex,
                           residue_segindex=[segids.index(x[2])
                                             for x in RESIDUES],
                           trajectory=True)
    u.add_TopologyAttr('names', names)
    u.add_TopologyAttr('resnames', [x[0] for x in RESIDUES])
    u.add_TopologyAttr('resids', [x[1] for x in RESIDUES])
    u.add_TopologyAttr('segids', segids)
    empty = np.array([], dtype=np.int32)
    return SimpleNamespace(
        universe=u, sel_idx=np.arange(len(names), dtype=np.int32),
        anions=empty, cations=empty, hb_acc=empty, met_acc=empty,
        xb_acc=empty, get_aro=lambda: tuple(range(16)))


def make_lambda_files(tmp_path, rows, n_ps=1000):
    """rows: (resname, state, resid, chain, coord_num, lambda value)."""
    lines = ['resname state# resid chain coordinateFile']
    for resname, state, resid, chain, num, value in rows:
        name = f'cphmd-coord-{num}.xvg'
        lines.append(f'{resname} {state} {resid} {chain or ""} {name}')
        time = np.arange(n_ps + 1, dtype=float)
        write_xvg(tmp_path / name, time, np.full(time.size, value))
    ref = tmp_path / 'lambdareference.dat'
    ref.write_text('\n'.join(lines) + '\n')
    return ref


def make_cphmd(tmp_path, rows, traj_frames=(0, 1)):
    ref = make_lambda_files(tmp_path, rows)
    return CpHMDManager(ref, tmp_path, '**/*-coord-*.xvg',
                        np.asarray(traj_frames), ps_per_frame=50)


def test_patch_adds_upper_bound_roles(tmp_path):
    cphmd = make_cphmd(tmp_path, [
        ('ASPT', 1, 52, 'PROA', 1, 0.9),
        ('HSPT', 1, 75, None, 2, 0.0),
        ('HSPT', 2, 75, None, 3, 1.0),
        ('HSPT', 3, 75, None, 4, 0.0),
        ('LYST', 1, 122, None, 5, 0.0),
    ])
    iman = make_iman()

    cphmd.patch_index_manager(iman)
    names = iman.universe.atoms.names

    assert sorted(names[iman.anions]) == ['OD1', 'OD2']
    assert sorted(names[iman.cations]) == ['ND1', 'NE2', 'NZ']
    assert sorted(names[iman.hb_acc]) == ['ND1', 'NE2', 'NZ']
    assert np.array_equal(iman.met_acc, iman.hb_acc)

    # HSPT75 is HSE: ND1 accepts, HD1 is absent; ASPT52:PROA deprotonated
    lookup_ = cphmd.build_atom_lookup(iman)
    atom_row, active = lookup_['acceptor']
    nd1 = np.flatnonzero(names == 'ND1')[0]
    assert active[atom_row[nd1]].all()
    atom_row, active = lookup_['proton']
    hd1 = np.flatnonzero(names == 'HD1')[0]
    hd2 = np.flatnonzero((names == 'HD2') & (iman.universe.atoms.segids ==
                                              'PROA'))[0]
    assert not active[atom_row[hd1]].any()
    assert not active[atom_row[hd2]].any()


def test_ambiguous_resid_needs_chain(tmp_path):
    cphmd = make_cphmd(tmp_path, [('ASPT', 1, 52, None, 1, 0.9)])

    with pytest.raises(ValueError, match='chain column'):
        cphmd.patch_index_manager(make_iman())


def test_lambda_alignment_uses_trajectory_frames(tmp_path):
    ref = tmp_path / 'lambdareference.dat'
    ref.write_text('resname state# resid chain coordinateFile\n'
                   'LYST 1 122 cphmd-coord-1.xvg\n')
    time = np.arange(0, 1001, dtype=float)
    write_xvg(tmp_path / 'cphmd-coord-1.xvg', time, (time >= 500) * 1.0)

    # start=10, stride=1 at 50 ps/frame -> frames at 500 and 550 ps
    cphmd = CpHMDManager(ref, tmp_path, '**/*-coord-*.xvg',
                         np.array([10, 11]), ps_per_frame=50)

    assert cphmd.sites[0]['roles']['cation']['NZ'].tolist() == [False, False]


def test_duplicate_xvg_files_raise(tmp_path):
    rows = [('LYST', 1, 122, None, 1, 0.0)]
    make_lambda_files(tmp_path, rows)
    (tmp_path / 'rep2').mkdir()
    make_lambda_files(tmp_path / 'rep2', rows)

    with pytest.raises(ValueError, match='several'):
        CpHMDManager(tmp_path / 'lambdareference.dat', tmp_path,
                     '**/*-coord-*.xvg', np.array([0]), ps_per_frame=50)


def test_missing_xvg_skips_residue(tmp_path, caplog):
    ref = make_lambda_files(tmp_path, [('LYST', 1, 122, None, 1, 0.0)])
    (tmp_path / 'cphmd-coord-1.xvg').unlink()

    with caplog.at_level(logging.WARNING, logger='InterMapLogger'):
        cphmd = CpHMDManager(ref, tmp_path, '**/*-coord-*.xvg',
                             np.array([0]), ps_per_frame=50)

    assert cphmd.sites == []
    assert 'missing .xvg' in caplog.text


# -- gating ------------------------------------------------------------------

def test_gate_chunk_suppresses_ionic_only():
    manager = make_manager()

    ijf_chunk = np.array([[1, 5, 0]], dtype=np.int32)
    inters_chunk = np.array([[True, True]], dtype=bool)
    gating_cols = gating(anion=(cols(0), empty_cols()))
    atom_lookup = lookup('anion', 1, [False])

    gated_ijf, gated_inters = manager.gate_chunk(
        ijf_chunk, inters_chunk, cols(0), gating_cols, atom_lookup)

    assert gated_ijf.shape == (1, 3)
    assert gated_inters.tolist() == [[False, True]]


def test_gate_chunk_ionic_depends_on_side():
    manager = make_manager()

    # Columns: 0 Cationic, 1 CationPi, 2 PiCation. Atom 3 is a neutral
    # cation in row 1 (rows 0) and row 2 (row 1).
    ijf_chunk = np.array([[3, 5, 0], [5, 3, 0]], dtype=np.int32)
    inters_chunk = np.ones((2, 3), dtype=bool)
    gating_cols = gating(cation=(cols(0, 1), cols(2)))
    atom_lookup = lookup('cation', 3, [False])

    _, gated_inters = manager.gate_chunk(
        ijf_chunk, inters_chunk, cols(0), gating_cols, atom_lookup)

    assert gated_inters.tolist() == [[False, False, True],
                                     [True, True, False]]


def test_gate_chunk_suppresses_missing_proton_on_both_sides():
    manager = make_manager()

    ijf_chunk = np.array([[7, 10, 0], [10, 7, 0], [10, 7, 1]],
                         dtype=np.int32)
    inters_chunk = np.array([[True, True], [True, True], [True, True]])
    gating_cols = gating(proton=(cols(0, 1), cols(0, 1)))
    atom_lookup = lookup('proton', 7, [False, True])

    gated_ijf, gated_inters = manager.gate_chunk(
        ijf_chunk, inters_chunk, cols(0, 1), gating_cols, atom_lookup)

    assert gated_ijf.tolist() == [[10, 7, 1]]
    assert gated_inters.tolist() == [[True, True]]


def test_gate_chunk_suppresses_acceptor_roles_on_both_sides():
    manager = make_manager()

    ijf_chunk = np.array([[9, 4, 0], [4, 9, 0]], dtype=np.int32)
    inters_chunk = np.array([
        [True, True, False, False],
        [False, False, True, True],
    ], dtype=bool)
    gating_cols = gating(acceptor=(cols(0, 1), cols(2, 3)))
    atom_lookup = lookup('acceptor', 9, [False])

    gated_ijf, gated_inters = manager.gate_chunk(
        ijf_chunk, inters_chunk, cols(0), gating_cols, atom_lookup)

    assert gated_ijf.shape == (0, 3)
    assert gated_inters.shape == (0, 4)


def test_gating_columns_per_role():
    gating_cols = CpHMDManager.get_gating_col_indices(
        ['PiCation', 'CationPi'], ['Cationic', 'HBDonor', 'HBAcceptor'])

    assert gating_cols['cation'][0].tolist() == [1, 2]
    assert gating_cols['cation'][1].tolist() == [0]
    assert gating_cols['acceptor'][0].tolist() == [4]
    assert gating_cols['acceptor'][1].tolist() == [3]
    assert gating_cols['proton'][0].tolist() == [0, 1, 2, 3, 4]


# -- config ------------------------------------------------------------------

@pytest.mark.parametrize('key, value, match', [
    ('lambda_ps_per_frame', '0', 'positive number'),
    ('hspt_states', 'HSP, HSD', 'hspt_states'),
    ('lambda_typo', '1', 'forbidden'),
])
def test_cphmd_config_validation(tmp_path, get_clean_cfg, write_modified_cfg,
                                 key, value, match):
    cfg = get_clean_cfg
    cfg['cphmd'] = {'lambda_ref': __file__, 'lambda_dir': str(tmp_path)}
    path = write_modified_cfg(tmp_path, cfg, 'cphmd', key, value)

    with pytest.raises((KeyError, ValueError), match=match):
        ConfigManager(cfg_path=path)


def test_cphmd_config_defaults(tmp_path, get_clean_cfg, write_modified_cfg):
    cfg = get_clean_cfg
    cfg['cphmd'] = {'lambda_ref': __file__, 'lambda_dir': str(tmp_path)}
    cfg['generals']['n_procs'] = '1'
    path = write_modified_cfg(tmp_path, cfg, 'cphmd', 'lambda_ps_per_frame',
                              '2.5')

    args = ConfigManager(cfg_path=path).config_args

    assert args['lambda_ps_per_frame'] == 2.5
    assert args['hspt_states'] == ('HSP', 'HSE', 'HSD')
    assert args['lambda_glob'] == '**/*-coord-*.xvg'


# -- end-to-end --------------------------------------------------------------

@pytest.fixture
def fix_numba_recursion():
    """The pipeline needs the real Numba KDTree (overrides conftest)."""
    yield


def test_workflow_gates_titratable_lysine(example_system_args, tmp_path,
                                          fix_numba_recursion):
    import copy

    from bitarray import util as bu

    from intermap.runner import workflow

    def run(name, **kwargs):
        args = copy.deepcopy(example_system_args)
        args.output_dir = str(tmp_path / name)
        (tmp_path / name).mkdir()
        for key, value in kwargs.items():
            setattr(args, key, value)
        results = workflow(args)
        return {tuple(x.replace('LYST_', 'LYS_') for x in k): bu.sc_decode(v)
                for k, v in results.items()}

    # LYS 11 of PROB salt-bridges the DNA; rename it as titratable
    pdb = tmp_path / 'cphmd.pdb'
    with open(example_system_args.topology) as f:
        lines = [line[:17] + 'LYST' + line[21:]
                 if line.startswith('ATOM') and line[17:21].strip() == 'LYS'
                 and int(line[22:26]) == 11 and line[72:76].strip() == 'PROB'
                 else line for line in f]
    pdb.write_text(''.join(lines))

    def run_cphmd(name, value):
        ref = make_lambda_files(tmp_path, [('LYST', 1, 11, 'PROB', 1, value)],
                                n_ps=100)
        return run(name, topology=str(pdb), lambda_ref=str(ref),
                   lambda_dir=str(tmp_path), lambda_glob='*-coord-*.xvg',
                   lambda_ps_per_frame=1.0)

    base = run('base')
    protonated = run_cphmd('protonated', 0.0)
    deprotonated = run_cphmd('deprotonated', 1.0)

    assert protonated == base
    removed = set(base) - set(deprotonated)
    lys = 'LYS_11_10_'  # resname_resid_resindex_ of LYS 11 in PROB
    assert removed and all(k[0].startswith(lys) for k in removed)
    assert {k[-1] for k in removed if k[0].startswith(lys + 'NZ')} == {
        'Cationic'}
    assert not any(k[0].startswith(lys + 'HZ3') for k in deprotonated)
    assert all(deprotonated[k] == base[k] for k in deprotonated)
