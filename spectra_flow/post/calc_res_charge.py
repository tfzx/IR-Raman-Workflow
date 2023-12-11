from typing import Optional, Sequence
import numpy as np
from encode_env import encode_by_env, decode_by_env

def calc_adj_mat1(structure: np.ndarray, pair_order: np.ndarray):
    _sturc = structure[0]
    _pair_order = pair_order[0]
    natoms = _sturc.shape[0]
    adj_mat = np.zeros((natoms, natoms))
    for i in range(natoms):
        for j in range(natoms):
            if _sturc[i, j]:
                if _pair_order[i, j] > 0:
                    adj_mat[i, j] = -1
                elif _pair_order[i, j] < 0:
                    adj_mat[i, j] = 1
    return adj_mat

def calc_adj_mat2(structure: np.ndarray, charges: np.ndarray):
    """
    `\sum_{j} adj[i][j] = charges[i]`
    """
    n = structure.shape[-1]
    assert np.allclose(np.sum(charges), 0)
    positive_edges = np.nonzero(np.triu(structure, k = 1))
    I_n = np.eye(n)
    incident_mat = I_n[:, positive_edges[0]] - I_n[:, positive_edges[1]]
    x, res, rank, s = np.linalg.lstsq(incident_mat, charges, rcond = None)
    # print(res, rank)
    adj_mat = np.zeros_like(structure, dtype = float)
    adj_mat[positive_edges] = x
    adj_mat -= adj_mat.T
    adj_mat = np.round(adj_mat, decimals = 2)
    assert np.allclose(adj_mat * (~structure), 0)
    assert np.allclose(np.sum(adj_mat, axis = -1), charges)
    return adj_mat

def get_residual_charge(atom_types: Sequence[int], valence_map: Sequence[float], 
                        wc_charge: np.ndarray, ions_charge: Optional[np.ndarray] = None):
    if ions_charge is None:
        ions_charge = 0
    atom_types = np.array(atom_types)
    valence_charge = np.zeros_like(atom_types, dtype = float)
    for i in range(len(valence_map)):
        valence_charge[atom_types == i] = valence_map[i]
    return valence_charge - ions_charge - wc_charge

def dump_charge_map(atom_types, residual_charge, structure):
    return encode_by_env(atom_types, structure, residual_charge)

def infer_charge(atom_types, charge_map, structure):
    return decode_by_env(atom_types, structure, charge_map)
