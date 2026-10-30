# Created by gonzalezroy at 3/11/25
import itertools as it

import numpy as np
import numpy_indexed as npi

from intermap.interactions import geometry as geom


def wb1(ijf_chunk, inters_chunk, waters, idxs, resconv, atomic=True):

    if not atomic:
        waters = resconv[waters]

    # Detect waters in i and j rows
    i_is_wat = geom.isin(ijf_chunk[:, 0], waters)
    j_is_wat = geom.isin(ijf_chunk[:, 1], waters)

    # Select the rows that have at least one water, that is connected by a hb
    a1_xor_a2 = i_is_wat ^ j_is_wat
    hb_inters = inters_chunk[:, idxs].any(axis=1)
    selected = a1_xor_a2 & hb_inters
    if not selected.any():
        return np.zeros((0, 4), dtype=np.int32)

    # Swap the rows to have the water in the first column (frames untouched)
    ijf_wat = ijf_chunk[selected]
    swap = j_is_wat[selected]
    ijf_wat[swap, 0], ijf_wat[swap, 1] = ijf_wat[swap, 1], ijf_wat[swap, 0]

    # In atomic resolution, identify each water molecule by its first atom so
    # that bridges through its different atoms (O, H) are grouped together
    if atomic:
        wat_res, first_atom = npi.group_by(resconv[waters]).min(waters)
        where = npi.indices(wat_res, resconv[ijf_wat[:, 0]])
        ijf_wat[:, 0] = first_atom[where]

    # Group the interactions by the water molecules
    groups = npi.group_by(ijf_wat[:, [0, 2]])
    sorter = groups.index.sorter
    slices = groups.index.slices
    indices = np.split(sorter, slices[1:-1])

    # Get the pairs of different residues bridged by the same water
    ijkf = []
    for index in indices:
        section = ijf_wat[index]
        k = section[0][0]
        f = section[0][2]
        for (v, j) in it.combinations(np.unique(section[:, 1]), 2):
            if atomic and resconv[v] == resconv[j]:
                continue
            ijkf.append((v, j, k, f))

    if not ijkf:
        return np.zeros((0, 4), dtype=np.int32)
    return np.asarray(ijkf, dtype=np.int32)
