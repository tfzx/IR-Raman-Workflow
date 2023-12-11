from typing import Optional, Sequence, Union
import numpy as np
from spectra_flow.utils import box_shift, get_distance, k_nearest_safe

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
            structure: np.ndarray, pair_order: np.ndarray, batch_size: int = 50):
    natoms = structure.shape[-1]
    # (..., num_wfc, 2)
    k_idx, k_dist = k_nearest_safe(wfc, coords, cells, k = 2, batch_size = batch_size)
    idx_2 = k_idx[..., 0] * natoms + k_idx[..., 1]
    _pair_order = pair_order.reshape(pair_order.shape[:-2] + (1, natoms ** 2))
    _structure = structure.reshape(structure.shape[:-2] + (1, natoms ** 2))
    
    # (..., num_wfc, 2, 3)
    neighbour_coords = np.take_along_axis(coords[..., None, :, :], k_idx[..., None], axis = -2)
    bond_length = np.linalg.norm(
        box_shift(neighbour_coords[..., 0, :] - neighbour_coords[..., 1, :], cells), 
        ord = 2, axis = -1
    )
    # (..., num_wfc)
    lone_pair = \
        (k_dist[..., 0]**2 + bond_length**2 <= k_dist[..., 1]**2) | \
        (k_dist[..., 1] > k_dist[..., 0]*4) | \
        np.take_along_axis(~_structure, idx_2[..., None], axis = -1).squeeze(-1)
    d_charge = np.take_along_axis(_pair_order, idx_2[..., None], axis = -1).squeeze(-1)
    d_charge[lone_pair] = 1
    # (..., num_wfc, 2)
    wfc_charge = np.stack([1 + d_charge, 1 - d_charge], axis = -1)

    # (..., num_wfc, 2, 3)
    delta_wfc = box_shift(wfc[..., None, :] - neighbour_coords, cells) * wfc_charge[..., None]
    delta_wfc = delta_wfc.reshape(delta_wfc.shape[:-3] + (-1, 3))   # (..., num_wfc * 2, 3)
    k_idx = k_idx.reshape(k_idx.shape[:-2] + (1, -1))               # (..., 1, num_wfc * 2)
    wfc_charge = wfc_charge.reshape(wfc_charge.shape[:-2] + (-1, 1)) # (..., num_wfc * 2, 1)
    atom_idx = np.arange(natoms).reshape([1] * (k_idx.ndim - 2) + [-1, 1])
    wc = np.matmul(k_idx == atom_idx, delta_wfc)
    charge_number = np.matmul(k_idx == atom_idx, wfc_charge)[..., 0]
    # wc = np.zeros_like(coords)
    # charge_number = np.zeros(coords.shape[:-1], dtype = int)
    # for ii in range(natoms):
    #     wc[..., ii, :] = np.matmul(k_idx == ii, delta_wfc)[..., 0, :]
    #     charge_number[..., ii] = np.matmul(k_idx == ii, wfc_charge)[..., 0, 0]
    return wc, charge_number


def filter_benzene(wfc: np.ndarray, coords: np.ndarray, cells: np.ndarray, 
                    benzene_idx, structure: np.ndarray, batch_size: int = 50):
    EPS = 1e-3
    check_close = lambda x, y: np.abs(x - y) < EPS
    benzene_coords = coords[..., benzene_idx, :]
    benzene_structure = structure[..., benzene_idx, :][..., :, benzene_idx]
    num_C = benzene_structure.shape[-1]
    # (..., num_wfc, 2)
    k_idx, k_dist = k_nearest_safe(wfc, benzene_coords, cells, k = 2, batch_size = batch_size)
    idx_2 = k_idx[..., 0] * num_C + k_idx[..., 1]
    _structure = benzene_structure.reshape(benzene_structure.shape[:-2] + (1, num_C ** 2))
    
    # (..., num_wfc, 2, 3)
    neighbour_coords = np.take_along_axis(benzene_coords[..., None, :, :], k_idx[..., None], axis = -2)
    bond_length = np.linalg.norm(
        box_shift(neighbour_coords[..., 0, :] - neighbour_coords[..., 1, :], cells), 
        ord = 2, axis = -1
    )
    # (..., num_wfc)
    benzene_wfc_mask = \
        (k_dist[..., 0] < 1.0) \
        & (k_dist[..., 0]**2 + bond_length**2 > k_dist[..., 1]**2) \
        & (k_dist[..., 1] < k_dist[..., 0] * 2) \
        & np.take_along_axis(_structure, idx_2[..., None], axis = -1).squeeze(-1)
    num_benzene_wfc = int(num_C / 2) * 3
    _num_wfc = np.sum(benzene_wfc_mask.astype(int), axis = -1)
    filter_mask = check_close(_num_wfc, num_benzene_wfc)
    benzene_wfc_mask = benzene_wfc_mask[filter_mask]

    # (..., num_benzene_wfc)
    benzene_shape = benzene_wfc_mask.shape[:-1] + (num_benzene_wfc, )
    benzene_wfc = wfc[filter_mask][benzene_wfc_mask, :].reshape(*benzene_shape, -1)
    k_idx = k_idx[filter_mask][benzene_wfc_mask, :].reshape(*benzene_shape, -1)
    wfc_double_bond = np.any(get_struc(benzene_wfc, cells[filter_mask], None, blmax = 0.8), axis = -1)
    _num_double_bond = np.sum(wfc_double_bond.astype(int), axis = -1)
    wfc_charge = wfc_double_bond * 1.5 + (~wfc_double_bond) * 3
    # (..., num_benzene_wfc, 2)
    wfc_charge = np.tile(wfc_charge[..., None], [1] * wfc_charge.ndim + [2]) / 2

    k_idx = k_idx.reshape(k_idx.shape[:-2] + (1, -1))               # (..., 1, num_wfc * 2)
    wfc_charge = wfc_charge.reshape(wfc_charge.shape[:-2] + (-1, 1)) # (..., num_wfc * 2, 1)
    atom_idx = np.arange(num_C).reshape([1] * (k_idx.ndim - 2) + [-1, 1])
    benzene_charge_number = np.matmul(k_idx == atom_idx, wfc_charge)[..., 0]
    filter_mask[filter_mask] &= check_close(_num_double_bond, num_benzene_wfc / 3 * 2) \
                                & np.all(check_close(benzene_charge_number, 3), axis = -1)
    return filter_mask


def calc_benzene_wc(wfc: np.ndarray, coords: np.ndarray, cells: np.ndarray, 
                    benzene_idx, structure: np.ndarray, pair_order: np.ndarray, batch_size: int = 50):
    benzene_coords = coords[..., benzene_idx, :]
    benzene_structure = structure[..., benzene_idx, :][..., :, benzene_idx]
    num_C = benzene_structure.shape[-1]
    # (..., num_wfc, 2)
    k_idx, k_dist = k_nearest_safe(wfc, benzene_coords, cells, k = 2, batch_size = batch_size)
    idx_2 = k_idx[..., 0] * num_C + k_idx[..., 1]
    _structure = benzene_structure.reshape(benzene_structure.shape[:-2] + (1, num_C ** 2))
    
    # (..., num_wfc, 2, 3)
    neighbour_coords = np.take_along_axis(benzene_coords[..., None, :, :], k_idx[..., None], axis = -2)
    bond_length = np.linalg.norm(
        box_shift(neighbour_coords[..., 0, :] - neighbour_coords[..., 1, :], cells), 
        ord = 2, axis = -1
    )
    # (..., num_wfc)
    benzene_wfc_mask = \
        (k_dist[..., 0] < 1.0) \
        & (k_dist[..., 0]**2 + bond_length**2 > k_dist[..., 1]**2) \
        & (k_dist[..., 1] < k_dist[..., 0] * 2) \
        & np.take_along_axis(_structure, idx_2[..., None], axis = -1).squeeze(-1)
    num_benzene_wfc = int(num_C / 2) * 3
    _num_wfc = np.sum(benzene_wfc_mask.astype(int), axis = -1)
    assert np.allclose(_num_wfc, num_benzene_wfc), _num_wfc

    # (..., num_benzene_wfc)
    benzene_shape = benzene_wfc_mask.shape[:-1] + (num_benzene_wfc, )
    benzene_wfc = wfc[benzene_wfc_mask, :].reshape(*benzene_shape, -1)
    neighbour_coords = neighbour_coords[benzene_wfc_mask, :, :].reshape(*benzene_shape, -1, 3)
    k_idx = k_idx[benzene_wfc_mask, :].reshape(*benzene_shape, -1)
    wfc_double_bond = np.any(get_struc(benzene_wfc, cells, None, blmax = 0.8), axis = -1)
    _num_double_bond = np.sum(wfc_double_bond.astype(int), axis = -1)
    assert np.allclose(_num_double_bond, num_benzene_wfc / 3 * 2), _num_double_bond
    wfc_charge = wfc_double_bond * 1.5 + (~wfc_double_bond) * 3
    # (..., num_benzene_wfc, 2)
    wfc_charge = np.tile(wfc_charge[..., None], [1] * wfc_charge.ndim + [2]) / 2

    # (..., num_benzene_wfc, 2, 3)
    delta_wfc = box_shift(benzene_wfc[..., None, :] - neighbour_coords, cells) * wfc_charge[..., None]
    delta_wfc = delta_wfc.reshape(delta_wfc.shape[:-3] + (-1, 3))   # (..., num_wfc * 2, 3)
    k_idx = k_idx.reshape(k_idx.shape[:-2] + (1, -1))               # (..., 1, num_wfc * 2)
    wfc_charge = wfc_charge.reshape(wfc_charge.shape[:-2] + (-1, 1)) # (..., num_wfc * 2, 1)
    atom_idx = np.arange(num_C).reshape([1] * (k_idx.ndim - 2) + [-1, 1])
    benzene_wc = np.matmul(k_idx == atom_idx, delta_wfc)
    benzene_charge_number = np.matmul(k_idx == atom_idx, wfc_charge)[..., 0]
    assert np.allclose(benzene_charge_number, 3), benzene_charge_number
    
    wc, charge_number = calc_wc(wfc[~benzene_wfc_mask, :].reshape(*wfc.shape[:-2], -1, 3), coords, cells, structure, pair_order)
    wc[..., benzene_idx, :] += benzene_wc
    charge_number[..., benzene_idx] += benzene_charge_number.astype(int)
    return wc, charge_number

def calc_ions_dipole0(coords: np.ndarray, cells: np.ndarray, adj_mat: np.ndarray):
    delta = box_shift(coords[..., None, :] - coords[..., None, :, :], cells) # type: ignore
    return np.sum(delta * adj_mat[..., None], axis = (-3, -2)) / 2

def calc_ions_dipole(coords: np.ndarray, cells: np.ndarray, adj_mat: np.ndarray, batch_size: int = 10):
    natoms = coords.shape[-2]
    dipole = np.zeros(list(coords.shape[:-2]) + [3], dtype = coords.dtype)
    for i in range(0, natoms, batch_size):
        end_i = min(natoms, i + batch_size)
        delta = box_shift(coords[..., i:end_i, None, :] - coords[..., None, :, :], cells) # type: ignore
        dipole += np.sum(delta * adj_mat[..., i:end_i, :, None], axis = (-3, -2))
    return dipole / 2

def calc_dipole(coords: np.ndarray, cells: np.ndarray, wc: np.ndarray, adj_mat: np.ndarray, batch_size: int = 10):
    return calc_ions_dipole(coords, cells, adj_mat, batch_size) - np.sum(wc, axis = -2)
    # return calc_ions_dipole0(coords, cells, adj_mat) - np.sum(wc, axis = -2)