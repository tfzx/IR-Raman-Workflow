from typing import Dict, List, Optional, Tuple, Union, IO
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

def recurcive_update(dict1: dict, dict2: dict):
    for key in dict2:
        if isinstance(dict2[key], dict):
            if key not in dict1:
                dict1[key] = {}
            if isinstance(dict1[key], dict):
                recurcive_update(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]

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

def conf_from_npz(raw_conf, type_map: Optional[List[str]] = None):
    conf_data = dict(raw_conf)
    types = conf_data["atom_types"]
    ntyp = np.max(types) + 1
    conf_data["atom_numbs"] = [int(np.sum(types == i)) for i in range(ntyp)]
    conf_data["atom_names"] = type_map if type_map is not None else [str(i) for i in range(ntyp)]
    conf_data["orig"] = np.array([0, 0, 0])
    return dpdata.System(data = conf_data)

def read_conf(conf_path: Path, conf_fmt: Dict[str, Union[List[str], str]]) -> dpdata.System:
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

def read_labeled(conf_path: Path, conf_fmt: Dict[str, Union[List[str], str]], label_name: str) -> Tuple[dpdata.System, np.ndarray]:
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
    confs = read_conf(conf_path, conf_fmt)
    try:
        label = np.loadtxt(conf_path / f"{label_name}.raw", dtype = float, ndmin = 2)
    except:
        label = np.load(conf_path / f"{label_name}.npy")
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

def k_nearest(coords_A: np.ndarray, coords_B: Optional[np.ndarray], cells: np.ndarray, k: int):
    """
        For each point in coords_A, choose the k-nearest points (in the box) among coords_B, and return the index.
        The distance is calculated in the sense of PBC.

        Parameters
        -------------
            coords_A (..., num_A, d): the coordinates of the central points. The size of the last axis is the dimension.
            coords_B (..., num_B, d): the coordinates of the points to be selected. 
            If B is None, A will be compared with itself, where the diagonal will be removed.
            box      (..., d): the PBC box. box[..., i] is the length of period along x_i.
            k: int, the number of the points selected from coords_B.

        Return
        -------------
            index (..., num_A, k): the index of the k-nearest points in coords_B.
    """
    self_comp = False
    if coords_B is None:
        coords_B = coords_A
        self_comp = True
    distance = np.linalg.norm(
        box_shift(
            coords_A[..., np.newaxis, :] - coords_B[..., np.newaxis, :, :],  # type: ignore
            cells[..., np.newaxis, np.newaxis, :, :]
        ), 
        ord = 2, axis = -1
    )
    if self_comp:
        write_to_diagonal(distance, np.inf, offset = 0, axis1 = -2, axis2 = -1)
    return np.argsort(distance, axis = -1)[..., :k]

def inv_cells(cells: np.ndarray):
    """
    Reciprocal cells.
    """
    inv_cells = np.zeros_like(cells)
    inv_cells[..., :, 0] = np.cross(cells[..., 1, :], cells[..., 2, :])
    inv_cells[..., :, 1] = np.cross(cells[..., 2, :], cells[..., 0, :])
    inv_cells[..., :, 2] = np.cross(cells[..., 0, :], cells[..., 1, :])
    vol = np.sum(inv_cells[..., :, 0] * cells[..., 0, :], axis = -1)
    inv_cells /= vol[..., np.newaxis, np.newaxis]
    return inv_cells

def to_frac(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """
    Transfer from the cartesian coordinate to fractional coordinate.
    """
    recip_cell = inv_cells(cells)
    return np.sum(coords[..., np.newaxis] * recip_cell, axis = -2)

def box_shift(dx: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """
    Shift the coordinates (dx) to the coordinates that have the smallest absolute value.
    """
    frac_c = to_frac(dx, cells)[..., np.newaxis]
    return dx - np.sum(np.round(frac_c) * cells, axis = -2)
    # nl = np.floor(np.sum(dx[..., np.newaxis] * recip_cell, axis = -2) * 2)[..., np.newaxis]
    # return dx - np.sum((nl + nl % 2) * cell / 2, axis = -2)

def do_pbc(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    '''
    Translate to the home cell.
    '''
    frac_c = to_frac(coords, cells)[..., np.newaxis]
    return coords - np.sum(np.floor(frac_c) * cells, axis = -2)


def _check_coords(coords: np.ndarray, cells: np.ndarray, eps: float) -> bool:
    delta = box_shift(coords[..., np.newaxis, :, :] - coords[..., np.newaxis, :], cells[..., np.newaxis, np.newaxis, :, :]) # type: ignore # type: ignore
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
    for ii in multiIndex(v1.shape[1:]):
        corr[:, ii] = np.convolve(v1[:, ii], v2[:, ii], 'valid')
    corr /= n
    return corr
'''

def calculate_corr(A: np.ndarray, B: np.ndarray, window: int, n: Optional[int] = None):
    """
    Calculate the correlation function: <A(0)*B(t)>. Here, A(t) and B(t) is a d-dimensional vector, 
    and A(t) * B(t) is the element-wise production, and the result is also a d-dimensional vector.

    Parameters
    -----
    A, B: np.ndarray, in shape of (num_t, d).
        The first dimension refers to the time, and the second dimension refers to the dimension of the vector.
    
    window: int.
        the width of window to approximate the ensemble average.
        <A(0)*B(t)> = 1 / window * \sum_{i = 0}^{window - 1} A(i)*B(t + i)

    n: int, Optional.
        Maximal time steps.
    
    Return
    -----
    corr: np.ndarray, in shape of (n, d).
        corr(t) = <A(0)*B(t)>
    """
    if A.ndim == 1 or B.ndim == 1:
        A = A.reshape(-1, 1)
        B = B.reshape(-1, 1)
    if n is None:
        n = min(A.shape[0], B.shape[0]) - window
    assert n <= min(A.shape[0], B.shape[0]), "The number of steps is too large!"
    v1 = np.concatenate([A[:n][::-1], np.zeros([window, A.shape[1]], dtype = np.float32)], axis = 0)
    v2 = B[:n + window]
    corr = np.fft.ifft(np.fft.fft(v1, axis = 0) * np.fft.fft(v2, axis = 0), axis = 0).real # type: ignore
    corr = corr[n - 1:n + window] / n
    return corr

def apply_gussian_filter(corr: np.ndarray, width: float):
    """
    Apply gaussian filter. Parameter width means the smoothing width.
    """
    nmax = corr.shape[0] - 1
    return corr * np.exp(-.5 * (0.5 * width * np.arange(nmax + 1) / nmax)**2)

def FILONC(DT: float, DOM: float, C: np.ndarray) -> np.ndarray:
    """
    Calculates the Fourier cosine transform by Filon's method.
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
    The same as FILONC while DOM = 2. * np.pi / tmax.
    This is implemented by FFT.
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
    CE = np.fft.fft(C[:-1:2]).real + 0.5 * (C[NMAX] - C[0]) # type: ignore
    CO = (np.fft.fft(C[1::2]) * np.exp(-THETA[:int(NMAX / 2)] * 1j)).real # type: ignore
    CE = np.concatenate([CE, CE, CE[0:1]])
    CO = np.concatenate([CO, -CO, CO[0:1]])
    CHAT = 2.0 * (BETA * CE + GAMMA * CO) * DT
    return CHAT

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
