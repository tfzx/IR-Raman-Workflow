import numpy as np
from copy import deepcopy

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

if __name__ == "__main__":
    p = {
        "a": {
            "11": 1,
            "22": 2
        },
        "b": [123],
        "c": {

        }
    }
    p_d = {
        "a": {
            "11": 2,
            "33": 3,
            "44": {
                "55": 5,
                "66": {
                    "77": 7
                }
            }
        },
        "b": {
            "123": 123
        },
        "c": {
            "123": 123
        }
    }
    p_copy = complete_by_default(p, p_d, True)
    print(p)
    print(p_copy)