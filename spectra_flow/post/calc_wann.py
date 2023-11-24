from typing import Optional, Sequence, Union
import numpy as np, dpdata
from spectra_flow.utils import box_shift, write_to_diagonal, get_distance

def k_nearest_safe(coords_A: np.ndarray, coords_B: Optional[np.ndarray], cells: np.ndarray, 
                   k: int, batch_size: int = -1):
    """
    For each atom in coords_A, choose the k-nearest atoms (in the cell) among coords_B, and return their indices.
    The distance is calculated in the sense of PBC.

    Parameters
    -------------
    coords_A (..., num_A, 3): 
    the coordinates of the central points. The size of the last axis is the dimension.

    coords_B (..., num_B, 3): 
    the coordinates of the points to be selected. 
    If B is None, A will be compared with itself, where the diagonal will be removed.

    cells (..., 3, 3): 
    the PBC box. cells[..., :, i] is the i-th axis of the cell.

    k: int, the number of the points selected from coords_B.

    batch_size: int, the batch size of atoms in coords_B at each time. 
    The required memory size is (..., num_A, k + batch_size).
    If batch_size <= 0, it will use the largest batch_size, 
    which means the required memory size is (..., num_A, num_B).

    Return
    -------------
    indices (..., num_A, k): the indices of the k-nearest points in coords_B.

    distances (..., num_A, k): the distances of the k-nearest points in coords_B.
    """
    self_comp = False
    if coords_B is None:
        coords_B = coords_A
        self_comp = True
    d = coords_B.shape[-2]
    k = min(d, k)
    batch_size = min(d - k, batch_size)
    if batch_size <= 0:
        distance = get_distance(coords_A, coords_B, cells, remove_diag = self_comp)
        k_index = np.argsort(distance, axis = -1)[..., :k]
        k_distance = np.take_along_axis(distance, k_index, axis = -1)
    else:
        _shape = list(coords_A.shape)
        _shape[-1] = k + batch_size
        k_index = np.empty(_shape, dtype = int)
        k_distance = np.empty(_shape, dtype = coords_B.dtype)
        k_index[..., :k] = np.arange(k)
        k_distance[..., :k] = get_distance(
            coords_A, coords_B[..., :k, :], cells, remove_diag = self_comp, offset = 0
        )
        for i in range(k, d, batch_size):
            end_i = min(d, i + batch_size)
            sz = end_i - i
            k_index[..., k:k + sz] = np.arange(i, end_i)
            k_distance[..., k:k + sz] = get_distance(
                coords_A, coords_B[..., i:end_i, :], cells, remove_diag = self_comp, offset = i
            )
            sort_idx = np.argsort(k_distance, axis = -1)
            k_index = np.take_along_axis(k_index, sort_idx, axis = -1)
            k_distance = np.take_along_axis(k_distance, sort_idx, axis = -1)
    return k_index[..., :k], k_distance[..., :k]


def get_struc(coords: np.ndarray, cells: np.ndarray, types: Sequence[float], blmax: Union[float, np.ndarray] = 1.6):
    distance = get_distance(coords, coords, cells, remove_diag = True)
    if isinstance(blmax, np.ndarray):
        natoms = coords.shape[-2]
        ntyp = blmax.shape[-1]
        types_arr = np.array(types)
        types_mat = types_arr.reshape(-1, 1) * ntyp + types_arr.reshape(1, -1)
        types_mat = types_mat.reshape(-1)
        blmax_mat = blmax.reshape(-1)[types_mat].reshape([1] * (coords.ndim - 2) + [natoms, natoms])
        structure = distance <= blmax_mat
    else:
        structure = distance <= blmax
    return structure

def get_share(types: Sequence[int], elec_neg: Sequence[float]):
    types_arr = np.array(types, dtype = int)
    natoms = types_arr.size
    ntypes = len(elec_neg)
    pair_order = np.zeros((natoms, natoms), dtype = int)
    for ii in range(ntypes):
        for jj in range(ntypes):
            mask = (types_arr == ii).reshape(-1, 1)*(types_arr == jj).reshape(1, -1)
            if np.allclose(elec_neg[ii], elec_neg[jj]):
                pair_order[mask] = 0
            elif elec_neg[ii] > elec_neg[jj]:
                pair_order[mask] = 1
            else:
                pair_order[mask] = -1
    return pair_order

def calc_wc(wfc: np.ndarray, coords: np.ndarray, cells: np.ndarray, 
            structure: np.ndarray, pair_order: np.ndarray):
    natoms = structure.shape[-1]
    k_idx, k_dist = k_nearest_safe(wfc, coords, cells, k = 2, batch_size = 10)    # (..., num_wfc, 2)
    idx_2 = k_idx[..., 0] * natoms + k_idx[..., 1]
    new_shape = list(pair_order.shape)
    new_shape[-2:] = 1, natoms ** 2
    _pair_order = pair_order.reshape(new_shape)
    _structure = structure.reshape(new_shape)
    
    neighbour_coords = np.take_along_axis(coords[..., None, :, :], k_idx[..., None], axis = -2)
    bond_length = np.linalg.norm(neighbour_coords[..., 0, :] - neighbour_coords[..., 1, :], axis = -1)
    lone_pair = \
        (k_dist[..., 0]**2 + bond_length**2 <= k_dist[..., 1]**2) | \
        (k_dist[..., 1] > k_dist[..., 0]*5) | \
        np.take_along_axis(~_structure, idx_2[..., None], axis = -1).squeeze(-1)
    share_wfc = (~lone_pair) & np.take_along_axis((_pair_order == 0), idx_2[..., None], axis = -1).squeeze(-1)
    share_wfc = share_wfc[..., None].astype(int)

    switch_mask = (~lone_pair) & np.take_along_axis((_pair_order < 0), idx_2[..., None], axis = -1).squeeze(-1)
    k_idx[switch_mask, :] = np.flip(k_idx[switch_mask, :], axis = -1)
    neighbour_coords[switch_mask, :, :] = np.flip(neighbour_coords[switch_mask, :, :], axis = -2)

    # (..., num_wfc, 2, 3)
    delta_wfc = box_shift(wfc[..., None, :] - neighbour_coords, 
                          cells[..., None, None, :, :])
    wc = np.zeros_like(coords)
    charge_number = np.zeros(coords.shape[:-1], dtype = int)
    for ii in range(natoms):
        wc[..., ii, :] = np.sum(delta_wfc[..., 0, :] * (k_idx[..., [0]] == ii) * (2 - share_wfc), axis = -2)\
                       + np.sum(delta_wfc[..., 1, :] * (k_idx[..., [1]] == ii) * share_wfc, axis = -2)
        charge_number[..., ii] = np.sum((k_idx[..., [0]] == ii) * (2 - share_wfc), axis = (-2, -1))\
                        + np.sum((k_idx[..., [1]] == ii) * share_wfc, axis = (-2, -1))
    return wc, charge_number

def calc_benzene_wc(wfc: np.ndarray, coords: np.ndarray, cells: np.ndarray, 
                    benzene_idx, structure: np.ndarray, pair_order: np.ndarray):
    benzene_coords = coords[..., benzene_idx, :]
    benzene_structure = structure[..., benzene_idx, :][..., :, benzene_idx]
    num_C = benzene_structure.shape[-1]
    k_idx, k_dist = k_nearest_safe(wfc, benzene_coords, cells, k = 2, batch_size = 10)    # (..., num_wfc, 2)
    idx_2 = k_idx[..., 0] * num_C + k_idx[..., 1]
    new_shape = list(benzene_structure.shape)
    new_shape[-2:] = 1, num_C ** 2
    _structure = benzene_structure.reshape(new_shape)
    
    neighbour_coords = np.take_along_axis(benzene_coords[..., None, :, :], k_idx[..., None], axis = -2)
    bond_length = np.linalg.norm(neighbour_coords[..., 0, :] - neighbour_coords[..., 1, :], axis = -1)
    lone_pair = \
        (k_dist[..., 0] > 2.0) | \
        (k_dist[..., 0]**2 + bond_length**2 <= k_dist[..., 1]**2) | \
        (k_dist[..., 1] > k_dist[..., 0]*2) | \
        np.take_along_axis(~_structure, idx_2[..., None], axis = -1).squeeze(-1)
    benzene_wfc_mask = (~lone_pair)
    num_benzene_wfc = int(num_C / 2) * 3
    assert np.all(np.sum(benzene_wfc_mask.astype(int), axis = -1) == num_benzene_wfc)

    benzene_shape = benzene_wfc_mask.shape[:-1] + (num_benzene_wfc, )
    benzene_wfc = wfc[benzene_wfc_mask, :].reshape(*benzene_shape, -1)
    neighbour_coords = neighbour_coords[benzene_wfc_mask, :, :].reshape(*benzene_shape, -1, 3)
    k_idx = k_idx[benzene_wfc_mask, :].reshape(*benzene_shape, -1)
    wfc_double_bond = np.any(get_struc(benzene_wfc, cells, None, blmax = 0.9), axis = -1)
    wfc_charge = wfc_double_bond * 1.5 + (~wfc_double_bond) * 3

    delta_wfc = box_shift(benzene_wfc[..., None, :] - neighbour_coords, 
                          cells[..., None, None, :, :]) * wfc_charge[..., None, None] / 2
    benzene_wc = np.zeros_like(benzene_coords)
    benzene_charge_number = np.zeros(benzene_coords.shape[:-1], dtype = float)
    for ii in range(num_C):
        benzene_wc[..., ii, :] = np.sum(delta_wfc * (k_idx[..., None] == ii) , axis = (-3, -2))
        benzene_charge_number[..., ii] = np.sum(wfc_charge[..., None] * (k_idx == ii), axis = (-2, -1)) / 2
    assert np.allclose(benzene_charge_number, 3), benzene_charge_number
    
    wc, charge_number = calc_wc(wfc[~benzene_wfc_mask, :].reshape(*wfc.shape[:-2], -1, 3), coords, cells, structure, pair_order)
    wc[..., benzene_idx, :] += benzene_wc
    charge_number[..., benzene_idx] += benzene_charge_number.astype(int)
    return wc, charge_number

def calc_ions_dipole0(coords: np.ndarray, cells: np.ndarray, adj_mat: np.ndarray):
    delta = box_shift(coords[..., None, :] - coords[..., None, :, :], 
                      cells[..., None, None, :, :]) # type: ignore
    return np.sum(delta * adj_mat[..., None], axis = (-3, -2)) / 2

def calc_ions_dipole(coords: np.ndarray, cells: np.ndarray, adj_mat: np.ndarray, batch_size: int = 10):
    natoms = coords.shape[-2]
    dipole = np.zeros(list(coords.shape[:-2]) + [3], dtype = coords.dtype)
    for i in range(0, natoms, batch_size):
        end_i = min(natoms, i + batch_size)
        delta = box_shift(coords[..., i:end_i, None, :] - coords[..., None, :, :], 
                cells[..., None, None, :, :]) # type: ignore
        dipole += np.sum(delta * adj_mat[..., i:end_i, :, None], axis = (-3, -2))
    return dipole / 2

def calc_dipole(coords: np.ndarray, cells: np.ndarray, wc: np.ndarray, adj_mat: np.ndarray, batch_size: int = 10):
    return calc_ions_dipole(coords, cells, adj_mat, batch_size) - np.sum(wc, axis = -2)
    # return calc_ions_dipole0(coords, cells, adj_mat) - np.sum(wc, axis = -2)