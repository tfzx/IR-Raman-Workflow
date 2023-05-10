from typing import Dict, List, Tuple, Union, IO
from tempfile import TemporaryFile
import numpy as np
from copy import deepcopy
import dpdata
from pathlib import Path
from dflow import executor
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

def conf_from_npz(raw_conf, type_map: List[str] = None):
    conf_data = dict(raw_conf)
    types = conf_data["atom_types"]
    ntyp = np.max(types) + 1
    conf_data["atom_numbs"] = [int(np.sum(types == i)) for i in range(ntyp)]
    conf_data["atom_names"] = type_map if type_map is not None else [str(i) for i in range(ntyp)]
    conf_data["orig"] = np.array([0, 0, 0])
    return dpdata.System(data = conf_data)

def read_conf(conf_path: Path, conf_fmt: Dict[str, Union[List[str], str]]) -> dpdata.System:
    fmt: str = conf_fmt.get("fmt", "")
    type_map = conf_fmt.get("type_map", None)
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
    confs = confs.sub_system(np.nonzero(mask)[0].tolist())
    tensor = tensor[mask, ...]
    return confs, tensor

'''
def calculate_corr(A: np.ndarray, B: np.ndarray, window: int, n: int):
    v1 = A[:n][::-1]
    v2 = B[:n + window]
    for ii in multiIndex(v1.shape[1:]):
        corr[:, ii] = np.convolve(v1[:, ii], v2[:, ii], 'valid')
    corr /= n
    return corr
'''

def calculate_corr(A: np.ndarray, B: np.ndarray, window: int, n: int = None):
    if A.ndim == 1 or B.ndim == 1:
        A = A.reshape(-1, 1)
        B = B.reshape(-1, 1)
    if not n:
        n = min(A.shape[0], B.shape[0]) - window
    assert n <= min(A.shape[0], B.shape[0]), "The number of steps is too large!"
    v1 = np.concatenate([A[:n][::-1], np.zeros([window, A.shape[1]], dtype = np.float32)], axis = 0)
    v2 = B[:n + window]
    corr = np.fft.ifft(np.fft.fft(v1, axis = 0) * np.fft.fft(v2, axis = 0), axis = 0).real
    corr = corr[n - 1:n + window] / n
    return corr

def apply_gussian_filter(corr: np.ndarray, width: int):
    nmax = corr.shape[0] - 1
    return corr * np.exp(-.5 * (0.5 * width * np.arange(nmax + 1) / nmax)**2)

def FILONC(DT: float, DOM: float, C: np.ndarray) -> np.ndarray:
    NMAX = C.shape[0] - 1
    if NMAX % 2 != 0:
        return
    TMAX = NMAX * DT
    NU = np.arange(NMAX + 1)
    OMEGA = NU * DOM
    THETA = OMEGA * DT
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
    def CAL_C0(theta: np.ndarray, args: Tuple[np.ndarray, np.ndarray]):
        a, b = args
        return np.dot(a, np.cos(theta * b))
    CE = np.apply_along_axis(CAL_C0, axis = 1, arr = THETA[:, np.newaxis], args = (C[::2], np.arange(0, NMAX + 1, 2)))
    CE -= 0.5 * (C[0] + C[NMAX] * np.cos(OMEGA * TMAX))
    CO = np.apply_along_axis(CAL_C0, axis = 1, arr = THETA[:, np.newaxis], args = (C[1::2], np.arange(1, NMAX, 2)))
    CHAT = 2.0 * ( ALPHA * C[NMAX] * np.sin ( OMEGA * TMAX ) + BETA * CE + GAMMA * CO ) * DT
    return CHAT

def FT(DT: float, C: np.ndarray) -> np.ndarray:
    """
    The same as FILONC while DOM = 2. * np.pi / tmax
    """
    NMAX = C.shape[0] - 1
    assert NMAX % 2 == 0, 'NMAX is not even!'
    DTH = 2 * np.pi / NMAX
    NU = np.arange(NMAX + 1)
    THETA = NU * DTH
    SINTH = np.sin(THETA)
    COSTH = np.cos(THETA)
    COSSQ = np.square(COSTH)
    THSQ  = np.square(THETA)
    THCUB = THSQ * THETA
    BETA  = 2. * ( THETA * ( 1. + COSSQ ) - 2. * SINTH * COSTH )
    GAMMA = 4. * ( SINTH - THETA * COSTH )
    BETA[0] = 2. / 3.
    GAMMA[0] = 4. / 3.
    BETA[1:] /= THCUB[1:]
    GAMMA[1:] /= THCUB[1:]
    CE = np.fft.fft(C[:-1:2]).real + 0.5 * (C[NMAX] - C[0])
    CO = (np.fft.fft(C[1::2]) * np.exp(-THETA[:int(NMAX / 2)] * 1j)).real
    CE = np.concatenate([CE, CE, CE[0:1]])
    CO = np.concatenate([CO, -CO, CO[0:1]])
    CHAT = 2.0 * (BETA * CE + GAMMA * CO) * DT
    return CHAT

def numerical_diff(y: np.ndarray, h: float):
    g = (y[2:] - y[:-2]) / (2 * h)
    g3, g4   = diff_8(g[:8])    # approx g[3], g[4]
    gn5, gn4 = diff_8(g[-8:])   # approx g[-4], g[-5]
    g[4] += (gn5 - g3) / 6.
    g[-5] += (g4 - gn4) / 6.
    g = g[4:-4]
    v = np.zeros((g.shape[0], 1))
    v[0] = -2; v[1] = 1; v[-1] = 1
    return np.fft.ifft(np.fft.fft(g, axis = 0) / (1 + np.fft.fft(v, axis = 0) / 6), axis = 0).real

def diff_8(g):
    w = np.array([336./2911., -1344./2911., 5040./2911., -1350./2911., 360./2911., -90./2911.])
    b = g[1:7].copy()
    b[0] -= g[0] / 6.
    b[-1] -= g[7] / 6.
    g_3 = np.dot(w, b)   # approx g[3]
    g_4 = np.dot(w[::-1], b)
    return g_3, g_4

def get_executor(exec_config: dict) -> executor:
    if exec_config["type"] == "bohrium":
        return DispatcherExecutor(machine_dict = {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "input_data": exec_config["params"],
            },
        },)


def _read_dump(f_dump: IO, f_cells: IO, f_coords: IO, f_types: IO, BUFFER: int = 50000):
    print("#### Read the lammps dump file ####")
    nAtoms = int(np.loadtxt(f_dump, dtype = int, skiprows = 3, max_rows = 1))
    print("Number of atoms =", nAtoms)
    types = np.loadtxt(f_dump, dtype = int, skiprows = 5, max_rows = nAtoms, usecols = [1]) - 1
    np.save(f_types, types)

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
        f_cells.close()
        f_coords.close()
        f_types.close()
    return sys
