import numpy as np
from copy import deepcopy
import dpdata

def kmesh(nx: int, ny: int, nz: int):
    kx = (np.arange(nx) / nx).reshape(-1, 1) * np.array([1, 0, 0])
    ky = (np.arange(ny) / ny).reshape(-1, 1) * np.array([0, 1, 0])
    kz = (np.arange(nz) / nz).reshape(-1, 1) * np.array([0, 0, 1])
    kpoints = kx.reshape(nx, 1, 1, 3) + ky.reshape(1, ny, 1, 3) + kz.reshape(1, 1, nz, 3)
    return np.concatenate([kpoints.reshape(-1, 3), np.ones((nx * ny * nz, 1), dtype = float) / (nx * ny * nz)], axis = 1)

def complete_by_default(params: dict, params_default: dict, if_copy: bool = False):
    if if_copy:
        params = deepcopy(params)
    for key in params_default:
        if isinstance(params_default[key], dict):
            if key not in params:
                params[key] = {}
            if isinstance(params[key], dict):
                complete_by_default(params[key], params_default[key], if_copy = False)
        else:
            if key not in params:
                params[key] = params_default[key]
    return params

def box_shift(dx: np.ndarray, box: np.ndarray):
    nl = np.floor(dx / (box / 2))
    return dx - (nl + (nl + 2) % 2) * box / 2

def _check_coords(coords: np.ndarray, box: np.ndarray, eps: float) -> bool:
    delta = box_shift(coords[..., np.newaxis, :, :] - coords[..., np.newaxis, :], box[..., np.newaxis, np.newaxis, :])
    mask = np.linalg.norm(delta, 2, axis = -1) < eps
    np.fill_diagonal(mask, False)
    return not mask.any()

def check_coords(coords: np.ndarray, box: np.ndarray, eps: float):
    c = np.concatenate([coords, box[..., np.newaxis, :]], axis = -2).reshape(coords.shape[0], -1)
    def check(arr: np.ndarray):
        arr = arr.reshape(-1, 3)
        c = arr[:-1]
        b = arr[-1]
        return _check_coords(c, b, eps)
    return np.apply_along_axis(check, axis = -1, arr = c)

def filter_confs(confs: dpdata.System, tensor: np.ndarray):
    mask = check_coords(confs["coords"], confs["cells"].diagonal(offset = 0, axis1 = 1, axis2 = 2), eps = 1e-3)
    confs = confs[mask]
    tensor = tensor[mask, ...]
    return confs, tensor
