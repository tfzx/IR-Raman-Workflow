from typing import Dict, List, Mapping, Optional, Tuple, Union, IO
from tempfile import TemporaryFile
import numpy as np
from copy import deepcopy
import dpdata, json
from pathlib import Path
from dflow.executor import Executor
from dflow.plugins.dispatcher import DispatcherExecutor

def kmesh(nx: int, ny: int, nz: int):
    kx = (np.arange(nx) / nx).reshape(-1, 1) * np.array([1, 0, 0])
    ky = (np.arange(ny) / ny).reshape(-1, 1) * np.array([0, 1, 0])
    kz = (np.arange(nz) / nz).reshape(-1, 1) * np.array([0, 0, 1])
    kpoints = kx.reshape(nx, 1, 1, 3) + ky.reshape(1, ny, 1, 3) + kz.reshape(1, 1, nz, 3)
    return np.concatenate([kpoints.reshape(-1, 3), np.ones((nx * ny * nz, 1), dtype = float) / (nx * ny * nz)], axis = 1)

def complete_by_default(params: dict, params_default: dict, if_copy: bool = False):
    if if_copy:
        params = deepcopy(params)
    for key, def_val in params_default.items():
        if isinstance(def_val, dict):
            if key not in params:
                params[key] = {}
            if isinstance(params[key], dict):
                complete_by_default(params[key], def_val, if_copy = False)
        else:
            if key not in params:
                params[key] = def_val
    return params

def recurcive_update(dict1: dict, dict2: dict):
    for key, val2 in dict2.items():
        if isinstance(val2, dict):
            if key not in dict1:
                dict1[key] = {}
            if isinstance(dict1[key], dict):
                recurcive_update(dict1[key], val2)
        else:
            dict1[key] = val2

def load_json(path: Union[str, Path]):
    with open(path, "r") as f:
        setting = json.load(f)
    return setting

def bohrium_login(account_config: Optional[dict] = None, debug: bool = False):
    from dflow import config, s3_config
    from dflow.plugins import bohrium
    from dflow.plugins.bohrium import TiefblueClient
    from getpass import getpass
    if debug:
        config["mode"] = "debug"
        return 
    config["host"] = "https://workflows.deepmodeling.com"
    config["k8s_api_server"] = "https://workflows.deepmodeling.com"
    config["dispatcher_image_pull_policy"] = "IfNotPresent"
    if account_config:
        bohrium.config.update(account_config)
    else:
        bohrium.config["username"] = getpass("Bohrium username: ")
        bohrium.config["password"] = getpass("Bohrium password: ")
        bohrium.config["project_id"] = getpass("Project ID: ")
    s3_config["repo_key"] = "oss-bohrium"
    s3_config["storage_client"] = TiefblueClient()

def conf_from_npz(raw_conf: Mapping[str, np.ndarray], type_map: Optional[List[str]] = None):
    conf_data = {"coords": raw_conf["coords"], "cells": raw_conf["cells"], "atom_types": raw_conf["atom_types"]}
    types = conf_data["atom_types"]
    try:
        type_map = raw_conf["type_map"].tolist()
    except:
        pass
    ntyp = np.max(types) + 1
    conf_data["atom_numbs"] = [int(np.sum(types == i)) for i in range(ntyp)]
    conf_data["atom_names"] = type_map if type_map is not None else [f"Type_{i}" for i in range(ntyp)]
    conf_data["orig"] = np.array([0, 0, 0])
    return dpdata.System(data = conf_data)

def read_conf(conf_path: Union[Path, str], conf_fmt: Dict[str, Union[List[str], str]]) -> dpdata.System:
    """
    Read confs by the format dict `conf_fmt`.

    Parameters
    -----
    conf_path: Path. The path to the confs.

    conf_fmt: dict.
        `{"fmt": "format, empty by default", "type_map": "None by default"}`
    """
    fmt: str = conf_fmt.get("fmt", "") # type: ignore
    type_map: List[str] = conf_fmt.get("type_map", None) # type: ignore
    fmt = fmt.strip()
    if fmt:
        head = fmt.split("/")[0]
        if head == "numpy":
            conf = conf_from_npz(np.load(conf_path), type_map)
        else:
            conf = dpdata.System(conf_path, fmt = fmt, type_map = type_map)
    else:
        try:
            conf = dpdata.System(conf_path, fmt = "deepmd/raw", type_map = type_map)
        except NotImplementedError:
            try:
                conf = dpdata.System(conf_path, fmt = "deepmd/npy", type_map = type_map)
            except NotImplementedError:
                try:
                    conf = dpdata.System(conf_path, type_map = type_map)
                except NotImplementedError:
                    conf = conf_from_npz(np.load(conf_path), type_map)
    return conf

def read_labeled(
        conf_path: Union[Path, List[Path]], 
        conf_fmt: Dict[str, Union[List[str], str]], 
        label_name: str
    ) -> Tuple[List[dpdata.System], List[np.ndarray]]:
    """
    Read labeled confs by the format dict `conf_fmt`. The label here means `dipole` or `polarizability`.

    Parameters
    -----
    conf_path: Path. The path to the confs.

    conf_fmt: dict.
        `{"fmt": "format, empty by default", "type_map": "None by default"}`.

    label_name: str. 
        The name of the label.
        Label will be read from `conf_path / "{label_name}.raw"` or `conf_path / "{label_name}.npy"`.
    
    Return
    -----
    confs, label: Tuple[dpdata.System, np.ndarray]
    """
    if isinstance(conf_path, Path):
        conf_path = [conf_path]
    elif isinstance(conf_path, str):
        conf_path = [Path(conf_path)]
    confs = []
    label = []
    for cp in conf_path:
        confs.append(read_conf(cp, conf_fmt))
        try:
            lb = np.loadtxt(cp / f"{label_name}.raw", dtype = float, ndmin = 2)
        except:
            lb = np.load(cp / f"{label_name}.npy")
        label.append(lb)
    return confs, label

def read_confs_list(conf_path_l: List[Path], conf_fmt: Dict[str, Union[List[str], str]]) -> dpdata.System:
    systems = []
    for conf_path in conf_path_l:
        systems.append(read_conf(conf_path, conf_fmt))
    return systems


def read_multi_sys(conf_path: Path, conf_fmt: Dict[str, Union[List[str], str]]) -> dpdata.MultiSystems:
    """
    Read multiSystems by the format dict `conf_fmt`.

    Parameters
    -----
    conf_path: Path. The path to the multiSystems.

    conf_fmt: dict.
        `{"fmt": "format, empty by default", "file_name": "'*' by default", "type_map": "None by default"}`
    """
    fmt: str = conf_fmt.get("fmt", "") # type: ignore
    fmt = fmt.strip()
    file_name = conf_fmt.get("file_name", "*")
    type_map: List[str] = conf_fmt.get("type_map", None) # type: ignore
    return dpdata.MultiSystems.from_dir(
        dir_name = conf_path[0], file_name = file_name, fmt = fmt, type_map = type_map
    )

def write_to_diagonal(a: np.ndarray, diag: Union[np.ndarray, float, int], offset: int = 0, axis1: int = 0, axis2: int = 1):
    diag_slices: List[Union[slice, list]] = [slice(None) for _ in a.shape]
    start_idx = [max(-offset, 0), max(offset, 0)]
    diag_len = min(a.shape[axis1] - start_idx[0], a.shape[axis2] - start_idx[1])
    assert diag_len >= 0
    if diag_len == 0:
        return
    diag_slices[axis1] = list(range(start_idx[0], start_idx[0] + diag_len))
    diag_slices[axis2] = list(range(start_idx[1], start_idx[1] + diag_len))
    a[tuple(diag_slices)] = diag

def get_distance(coords_A: np.ndarray, coords_B: Optional[np.ndarray], cells: np.ndarray, 
                 remove_diag: bool = False, offset: int = 0):
    """
        Calculate the distances between coords_A and coords_B.
        The distance is calculated in the sense of PBC.

        Parameters
        -------------
            coords_A (..., num_A, 3): the coordinates of the central points. The size of the last axis is the dimension.
            coords_B (..., num_B, 3): the coordinates of the points to be selected. 
            If B is None, A will be compared with itself.
            cells    (..., 3, 3): the PBC cells.
            remove_diag, bool: whether to fill the diagonal with np.inf.

        Return
        -------------
            distance (..., num_A, num_B): the matrix of distances.
    """
    if coords_B is None:
        coords_B = coords_A
    distance = np.linalg.norm(
        box_shift(
            coords_A[..., None, :] - coords_B[..., None, :, :],  # type: ignore
            cells
        ), 
        ord = 2, axis = -1
    )
    if remove_diag:
        write_to_diagonal(distance, np.inf, offset = offset, axis1 = -2, axis2 = -1)
    return distance

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
        k_index = np.argpartition(distance, k, axis = -1)[..., :k]
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
            partition_idx = np.argpartition(k_distance, k, axis = -1)
            k_index = np.take_along_axis(k_index, partition_idx, axis = -1)
            k_distance = np.take_along_axis(k_distance, partition_idx, axis = -1)
    sort_idx = np.argsort(k_distance[..., :k], axis = -1)
    k_index = np.take_along_axis(k_index[..., :k], sort_idx, axis = -1)
    k_distance = np.take_along_axis(k_distance[..., :k], sort_idx, axis = -1)
    return k_index, k_distance

def k_nearest(coords_A: np.ndarray, coords_B: Optional[np.ndarray], cells: np.ndarray, k: int):
    """
        For each atom in coords_A, choose the k-nearest atoms (in the box) among coords_B, and return their index.
        The distance is calculated in the sense of PBC.

        Parameters
        -------------
            coords_A (..., num_A, 3): the coordinates of the central points. The size of the last axis is the dimension.
            coords_B (..., num_B, 3): the coordinates of the points to be selected. 
            If B is None, A will be compared with itself, where the diagonal will be removed.
            cells    (..., 3, 3): the PBC box. box[..., i] is the length of period along x_i.
            k: int, the number of the points selected from coords_B.

        Return
        -------------
            index (..., num_A, k): the index of the k-nearest points in coords_B.
    """
    self_comp = False
    if coords_B is None:
        coords_B = coords_A
        self_comp = True
    # distance = np.linalg.norm(
    #     box_shift(
    #         coords_A[..., np.newaxis, :] - coords_B[..., np.newaxis, :, :],  # type: ignore
    #         cells[..., np.newaxis, np.newaxis, :, :]
    #     ), 
    #     ord = 2, axis = -1
    # )
    # if self_comp:
    #     write_to_diagonal(distance, np.inf, offset = 0, axis1 = -2, axis2 = -1)
    distance = get_distance(coords_A, coords_B, cells, remove_diag = self_comp)
    k_index = np.argsort(distance, axis = -1)[..., :k]
    k_distance = np.take_along_axis(distance, k_index, axis = -1)
    return k_index, k_distance

def inv_cells(cells: np.ndarray):
    """
    Reciprocal cells.
    """
    inv_cells = np.linalg.inv(cells)
    # inv_cells = np.zeros_like(cells)
    # inv_cells[..., :, 0] = np.cross(cells[..., 1, :], cells[..., 2, :])
    # inv_cells[..., :, 1] = np.cross(cells[..., 2, :], cells[..., 0, :])
    # inv_cells[..., :, 2] = np.cross(cells[..., 0, :], cells[..., 1, :])
    # vol = np.sum(inv_cells[..., :, 0] * cells[..., 0, :], axis = -1)
    # inv_cells /= vol[..., np.newaxis, np.newaxis]
    return inv_cells

def _coords_cells_mul(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    if coords.ndim >= cells.ndim:
        d0 = coords.ndim - cells.ndim + 1
        _shape = coords.shape
        return np.matmul(coords.reshape(_shape[:-d0-1] + (-1, 3)), cells).reshape(_shape)
    else:
        return np.matmul(coords[..., None, :], cells).squeeze(-2)

def to_frac(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """
    Transfer from the cartesian coordinate to fractional coordinate.

    Parameters
    -----
    coords: np.ndarray,
    in shape of (..., 3)

    cells: np.ndarray,
    in shape of (..., 3, 3)

    Return
    -----
    fractional coords: np.ndarray,
    in shape of (..., 3)
    """
    recip_cell = inv_cells(cells)
    return _coords_cells_mul(coords, recip_cell)

def box_shift(dx: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """
    Shift the coordinates (dx) to the coordinates that have the smallest absolute value.

    Parameters
    -----
    dx: np.ndarray,
    in shape of (..., 3)

    cells: np.ndarray,
    in shape of (..., 3, 3)

    Return
    -----
    shifted_dx: np.ndarray,
    in shape of (..., 3)
    """
    # frac_c = to_frac(dx, cells)[..., np.newaxis]            # (..., 3, 1)
    return dx - _coords_cells_mul(np.round(to_frac(dx, cells)), cells)
    # return dx - np.sum(np.round(frac_c) * cells, axis = -2) # (..., 3)

def do_pbc(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    '''
    Translate to the home cell.

    Parameters
    -----
    coords: np.ndarray,
    in shape of (..., natom, 3)

    cells: np.ndarray,
    in shape of (..., 3, 3)

    Return
    -----
    translated coords: np.ndarray,
    in shape of (..., 3)
    '''
    # _cells = cells[..., np.newaxis, :, :]       # TODO
    # frac_c = to_frac(coords, _cells)[..., np.newaxis]
    return coords - _coords_cells_mul(np.floor(to_frac(coords, cells)), cells)


def _check_coords(coords: np.ndarray, cells: np.ndarray, eps: float) -> bool:
    delta = box_shift(coords[..., np.newaxis, :, :] - coords[..., np.newaxis, :], cells) # type: ignore # type: ignore
    mask = np.linalg.norm(delta, 2, axis = -1) < eps
    np.fill_diagonal(mask, False)
    return not mask.any()

def check_coords(coords: np.ndarray, cells: np.ndarray, eps: float):
    c = np.concatenate([coords, cells], axis = -2).reshape(coords.shape[0], -1)
    def check(arr: np.ndarray):
        arr = arr.reshape(-1, 3)
        c = arr[:-3]
        b = arr[-3:]
        return _check_coords(c, b, eps)
    return np.apply_along_axis(check, axis = -1, arr = c)

def filter_confs(confs: dpdata.System, tensor: Optional[np.ndarray] = None):
    """
    Filter the confs to remove the conf that some atoms are too close.
    """
    mask = check_coords(confs["coords"], confs["cells"], eps = 1e-3) # type: ignore
    confs = confs.sub_system(np.nonzero(mask)[0].tolist())
    if tensor is not None:
        tensor = tensor[mask, ...]
        return confs, tensor
    else:
        return confs

'''
def calculate_corr(A: np.ndarray, B: np.ndarray, window: int, n: int):
    v1 = A[:n][::-1]
    v2 = B[:n + window]
    corr = np.empty((window + 1, v1.shape[1]), dtype = float)
    for ii in range(v1.shape[1]):
        corr[:, ii] = np.convolve(v1[:, ii], v2[:, ii], 'valid')
    corr /= n
    return corr
'''

def calculate_corr(A: np.ndarray, B: np.ndarray, NMAX: int, window: Optional[int] = None):
    """
    Calculate the correlation function: `corr(t) = <A(0) * B(t)>`. 
    Here, `A(t)` and `B(t)` are arrays of the same dimensions, 
    and `A(t) * B(t)` is the element-wise multiplication. 
    The esenmble average `< >` is estimated by moving average.

    Parameters
    -----
    A, B: np.ndarray, in shape of (num_t, ...).
        The first dimension refers to the time steps, and its size can be different.
        The remaining dimensions (if present) must be the same.
    
    NMAX: int.
        Maximal time steps. Calculate `corr(t)` with `0 <= t <= NMAX`.
    
    window: int, optional.
        The width of window to do the moving average. 

        `<A(0) * B(t)> = 1 / window * \sum_{i = 0}^{window - 1} A(i) * B(t + i)`. 

    Return
    -----
    corr: np.ndarray, in shape of (NMAX + 1, ...).
        `corr(t) = <A(0) * B(t)>`
    """
    if A.ndim == 1 or B.ndim == 1:
        A = A.reshape(-1, 1)
        B = B.reshape(-1, 1)
    if window is None:
        window = min(A.shape[0], B.shape[0] - NMAX)
    # Prepare for convolution
    v1 = A[:window][::-1]; v2 = B[:window + NMAX]
    pad_width = [(0, 0)] * A.ndim
    pad_width[0] = (0, NMAX)
    v1 = np.pad(v1, pad_width, "constant", constant_values = 0)
    # Convolve by FFT
    corr = np.fft.ifft(np.fft.fft(v1, axis = 0) * np.fft.fft(v2, axis = 0), axis = 0).real # type: ignore
    # Moving average
    corr = corr[window - 1:window + NMAX] / window
    return corr

def apply_gussian_filter(corr: np.ndarray, width: float):
    """
    Apply gaussian filter. Parameter `width` means the smoothing width.
    """
    nmax = corr.shape[0] - 1
    return corr * np.exp(-.5 * (0.5 * width * np.arange(nmax + 1) / nmax)**2)

def apply_lorenz_filter(corr: np.ndarray, width: float, dt):
    """
    Apply Cauchy-Lorenz filter. Parameter `width` means the smoothing width.
    """
    nmax = corr.shape[0] - 1
    b = width * 2.99792458e-3
    M = int(1 / (dt * 0.01 * 2)) * 2
    M = max(M, nmax)
    dx = 1 / (M * dt)
    NX = int(50 * np.sqrt(b) / dx / 2) * 2
    x = np.arange(NX + 1) * dx
    p = b / (b**2 + x**2)
    _, ph = FT(dx, p, M)
    return corr * ph[:nmax + 1]

def FILONC(DT: float, DOM: float, C: np.ndarray, M: Optional[int] = None) -> np.ndarray:
    """
    Calculate the Fourier cosine transform by Filon's method.
    A correlation function, C(t), in the time domain, is
    transformed to a spectrum CHAT(OMEGA) in the frequency domain.

    Usage:

    The routine requires that the number of intervals, nmax, is
    even and checks for this condition. The first value of c(t)
    is at t = 0. The maximum time for the correlation function is
    tmax = dt * nmax. For an accurate transform c(tmax)=0.

    Parameters
    ------
    C: ndarray, the correlation function.
    DT: float, time interval between points in C.
    DOM: float, frequency interval for CHAT.
    M: Optional[int], number of intervals on the frequency axis.
    `M = NMAX` by default.

    Return
    -----
    CHAT: ndarray, the 1-d cosine transform.

    Reference:
    -----
    FILON, proc roy soc edin, 49 38, 1928.
    """
    """
    Principal variables:
    NMAX: int,
        number of intervals on the time axis
    OMEGA: ndarray[float],
        the frequency
    TMAX: float,
        maximum time in corrl. function
    ALPHA, BETA, GAMMA: ndarray[float],
        filon parameters
    NU: ndarray[int],
        frequency index
    """
    NMAX = C.shape[0] - 1
    assert NMAX % 2 == 0
    TMAX = NMAX * DT
    if M is None:
        M = NMAX
    elif M % 2 != 0:
        M += 1
    NU = np.arange(M + 1)
    OMEGA = NU * DOM
    THETA = OMEGA * DT
    # CALCULATE THE FILON PARAMETERS
    ALPHA, BETA, GAMMA = _FILON_PARAMS(THETA)
    def CAL_C0(theta: np.ndarray, args: Tuple[np.ndarray, np.ndarray]):
        a, b = args
        return np.dot(a, np.cos(theta * b))
    # DO THE SUM OVER THE EVEN ORDINATES
    # CE[k] = SUM_{i even} C(i) * COS ( THETA[k] * i ) - (C(0) + C(NMAX) * COS ( OMEGA[k] * TMAX ))
    CE = np.apply_along_axis(CAL_C0, axis = 1, arr = THETA[:, np.newaxis], args = (C[::2], np.arange(0, NMAX + 1, 2)))
    CE -= 0.5 * (C[0] + C[NMAX] * np.cos(OMEGA * TMAX))
    # DO THE SUM OVER THE ODD ORDINATES
    # CO[k] = SUM_{i odd} C(i) * COS ( THETA[k] * i )
    CO = np.apply_along_axis(CAL_C0, axis = 1, arr = THETA[:, np.newaxis], args = (C[1::2], np.arange(1, NMAX, 2)))
    CHAT = 2.0 * ( ALPHA * C[NMAX] * np.sin ( OMEGA * TMAX ) + BETA * CE + GAMMA * CO ) * DT
    return CHAT

def FT(DT: float, C: np.ndarray, M: Optional[int] = None) -> np.ndarray:
    """
    Perform a cosine transform on the correlation function using FFT.
    The same as FILONC while `DOM = 2\pi / (M * DT)` (or `OMEGA_MAX = 2\pi / DT`).

    Parameters
    -----
    C: ndarray
        the correlation function.
    DT: float
        time interval between points in C.
    M: Optional[int]
        number of intervals on the frequency axis. Default is `len(corr) - 1`.

    Returns
    -----
    Frequency and the 1-d cosine transform of the correlation function.
    freq: float, frequency. `freq = 1 / (M * DT)` 
    CHAT: np.ndarray, the 1-d cosine transform.
    """
    NMAX = C.shape[0] - 1
    if NMAX % 2 != 0:
        raise ValueError("NMAX (=len(C)-1) must be even for the cosine transform.")
    if M is None:
        M = NMAX
    elif M % 2 != 0:
        M += 1
        
    freq = 1 / (M * DT)
    DTH = 2 * np.pi / M
    NU = np.arange(M + 1)
    THETA = NU * DTH

    ALPHA, BETA, GAMMA = _FILON_PARAMS(THETA)
    CE, CO = _FFT_OE(C, M)
    CE, CO = CE.real, CO.real
    CE -= 0.5 * (C[0] + C[NMAX] * np.cos(THETA * NMAX))

    CHAT = 2.0 * (ALPHA * C[NMAX] * np.sin (THETA * NMAX) + BETA * CE + GAMMA * CO) * DT
    return freq, CHAT

def FT_sin(DT: float, C: np.ndarray, M: Optional[int] = None) -> np.ndarray:
    """
    Perform a sine transform on the correlation function using FFT.

    Parameters
    -----
    C: ndarray
        the correlation function.
    DT: float
        time interval between points in C.
    M: Optional[int]
        number of intervals on the frequency axis. Default is `len(corr) - 1`.

    Returns
    -----
    Frequency and the 1-d sine transform of the correlation function.
    freq: float, frequency. `freq = 1 / (M * DT)` 
    CHAT: np.ndarray, the 1-d sine transform.
    """
    NMAX = C.shape[0] - 1
    if NMAX % 2 != 0:
        raise ValueError("NMAX (=len(C)-1) must be even for the sine transform.")
    if M is None:
        M = NMAX
    elif M % 2 != 0:
        M += 1
    
    freq = 1 / (M * DT)
    DTH = 2 * np.pi / M
    NU = np.arange(M + 1)
    THETA = NU * DTH

    ALPHA, BETA, GAMMA = _FILON_PARAMS(THETA)
    CE, CO = _FFT_OE(C, M)
    CE, CO = CE.imag, CO.imag
    CE -= 0.5 * (C[NMAX] * np.sin(THETA * NMAX))

    CHAT = 2.0 * (ALPHA * (C[0] - C[NMAX] * np.cos(THETA * NMAX)) + BETA * CE + GAMMA * CO) * DT
    return freq, CHAT

def _FILON_PARAMS(THETA: np.ndarray) -> np.ndarray:
    """
    Calculate the filon parameters.
    """
    SINTH = np.sin(THETA)
    COSTH = np.cos(THETA)
    SINSQ = np.square(SINTH)
    COSSQ = np.square(COSTH)
    THSQ  = np.square(THETA)
    THCUB = THSQ * THETA
    ALPHA = 1. * ( THSQ + THETA * SINTH * COSTH - 2. * SINSQ )
    BETA  = 2. * ( THETA * ( 1. + COSSQ ) - 2. * SINTH * COSTH )
    GAMMA = 4. * ( SINTH - THETA * COSTH )
    ALPHA[0] = 0.
    BETA[0] = 2. / 3.
    GAMMA[0] = 4. / 3.
    ALPHA[1:] /= THCUB[1:]
    BETA[1:] /= THCUB[1:]
    GAMMA[1:] /= THCUB[1:]
    return ALPHA, BETA, GAMMA

def _FFT_OE(C: np.ndarray, M: int):
    M0 = int(M / 2)
    DTH = 2 * np.pi / M

    # Even coordinates
    CE = _range_fft(C[::2], M0) # type: ignore
    CE = np.concatenate([CE, CE, CE[0:1]])

    # Odd coordinates
    CO = _range_fft(C[1::2], M0) * np.exp(-np.arange(M0) * DTH * 1j) # type: ignore
    CO = np.concatenate([CO, -CO, CO[0:1]])
    return CE, CO

def _range_fft(a: np.ndarray, n: Optional[int] = None, axis: int = -1):
    """
    Compute `a_hat[..., l, ...] = \sum_{k=1}^{a.shape[axis]} a[..., k, ...]e^{-(2kl\pi/n)}`
    """
    axis %= a.ndim
    l = a.shape[axis]
    if n is None:
        n = l
    if n >= l:
        return np.fft.fft(a, n, axis)
    num_n = int(l / n)
    l0 = n * num_n
    new_shape = list(a.shape)
    new_shape[axis] = n
    new_shape.insert(axis, num_n)
    a_main = np.sum(a.take(range(l0), axis).reshape(new_shape), axis)
    a_tail = a.take(range(l0, l), axis)
    return np.fft.fft(a_main, n, axis) + np.fft.fft(a_tail, n, axis)

def numerical_diff(y: np.ndarray, h: float):
    g = (y[2:] - y[:-2]) / (2 * h) # type: ignore
    g3, g4   = diff_8(g[:8])    # approx g[3], g[4]
    gn5, gn4 = diff_8(g[-8:])   # approx g[-4], g[-5]
    g[4] += (gn5 - g3) / 6.
    g[-5] += (g4 - gn4) / 6.
    g = g[4:-4]
    v = np.zeros((g.shape[0], 1))
    v[0] = -2; v[1] = 1; v[-1] = 1
    return np.fft.ifft(np.fft.fft(g, axis = 0) / (1 + np.fft.fft(v, axis = 0) / 6), axis = 0).real # type: ignore

def diff_8(g):
    w = np.array([336./2911., -1344./2911., 5040./2911., -1350./2911., 360./2911., -90./2911.])
    b = g[1:7].copy()
    b[0] -= g[0] / 6.
    b[-1] -= g[7] / 6.
    g_3 = np.dot(w, b)   # approx g[3]
    g_4 = np.dot(w[::-1], b)
    return g_3, g_4

def get_executor(exec_config: dict) -> Executor:
    """
    Get executor. Only support bohrium now.
    """
    if exec_config["type"] == "bohrium":
        return DispatcherExecutor(machine_dict = {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "input_data": exec_config["params"],
            },
        },)
    else:
        raise NotImplementedError(f"Unknown executor type: {exec_config['type']}")


def _read_dump(f_dump: IO, f_cells: IO, f_coords: IO, f_types: IO, BUFFER: int = 50000):
    print("#### Read the lammps dump file ####")
    nAtoms = int(np.loadtxt(f_dump, dtype = int, skiprows = 3, max_rows = 1))
    print("Number of atoms =", nAtoms)
    types = np.loadtxt(f_dump, dtype = int, skiprows = 5, max_rows = nAtoms, usecols = [1]) - 1
    np.save(f_types, types) # type: ignore

    f_dump.seek(0, 0)
    step = 0
    idx_buffer = 0
    box_buffer = np.zeros((BUFFER, 9), dtype = float)
    coords_buffer = np.zeros((BUFFER, nAtoms * 3), dtype = float)
    while f_dump.readline() != "":
        box = np.loadtxt(f_dump, dtype = float, skiprows = 4, max_rows = 3, usecols = [1])
        if box.size == 0:
            break
        box_buffer[idx_buffer] = np.diag(box.reshape(-1)).reshape(-1)
        coords = np.loadtxt(f_dump, dtype = float, skiprows = 1, max_rows = nAtoms, usecols = [2, 3, 4])
        if coords.size == 0:
            print("[Warning]: cannot get the coordinates info!")
            break
        coords_buffer[idx_buffer] = coords.reshape(-1)
        idx_buffer += 1
        if idx_buffer >= BUFFER:
            np.savetxt(f_cells, box_buffer)
            np.savetxt(f_coords, coords_buffer)
            idx_buffer = 0
        if step % 10000 == 0:
            print("current step:", step)
        step += 1
    if idx_buffer > 0:
        np.savetxt(f_cells, box_buffer[:idx_buffer])
        np.savetxt(f_coords, coords_buffer[:idx_buffer])
    return step

def read_lmp_dump(dump_file: Path, BUFFER: int = 50000) -> Dict[str, np.ndarray]:
    """
    Read the lammps/dump file line by line. This is for cases when dump file is too large to totally read into RAM.
    """
    try:
        f_cells = TemporaryFile('r+')
        f_coords = TemporaryFile('r+')
        f_types = TemporaryFile()
        with open(dump_file, 'r') as f:
            _read_dump(f, f_cells, f_coords, f_types, BUFFER)
        sys = {}
        f_coords.seek(0, 0)
        sys["coords"] = np.loadtxt(f_coords, ndmin = 2)
        sys["coords"] = sys["coords"].reshape(sys["coords"].shape[0], -1, 3)
        f_cells.seek(0, 0)
        sys["cells"] = np.loadtxt(f_cells, ndmin = 2).reshape(-1, 3, 3)
        f_types.seek(0, 0)
        sys["atom_types"] = np.load(f_types).reshape(-1)
    finally:
        f_cells.close() # type: ignore
        f_coords.close() # type: ignore
        f_types.close() # type: ignore
    return sys

def dump_to_fmt(name: Union[str, Path], confs: Optional[dpdata.System], fmt: str, *args, in_fmt: Optional[dict] = None, **kwargs):
    """
    Dump the system (or configurations) into the specified format, and return the format dict.
    """
    if isinstance(name, str):
        conf_path = Path(name)
    else:
        conf_path = name
    if confs is None:
        conf_path.touch()
    else:
        confs.to(fmt, conf_path, *args, **kwargs)
    conf_fmt = {"fmt": fmt}
    if in_fmt is not None and "type_map" in in_fmt:
        conf_fmt["type_map"] = in_fmt["type_map"]
    return conf_path, conf_fmt
