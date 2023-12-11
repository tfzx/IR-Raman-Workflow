from typing import List, Sequence, Union
import numpy as np
from collections import namedtuple

EnvState = namedtuple("EnvState", ["EnvDataMap", "EnvTypeMap"])

def build_next_env(env_types: np.ndarray, structure: List[np.ndarray]):
    next_type_dict = {}
    next_type_map = []
    next_types = []
    start_idx = 0
    for ii in range(len(structure)):
        _struc = structure[ii]
        _env_types = env_types[start_idx:start_idx + _struc.shape[0]]
        start_idx += _struc.shape[0]
        for jj in range(_struc.shape[0]):
            _type = tuple([_env_types[jj]] + np.sort(_env_types[_struc[jj]]).tolist())
            if _type not in next_type_dict:
                next_type_map.append(_type)
                next_type_dict[_type] = len(next_type_dict)
            next_types.append(next_type_dict[_type])
    return np.array(next_types, dtype = int), tuple(next_type_map)

def get_next_env(env_types: np.ndarray, type_map: dict, structure: np.ndarray):
    next_types = []
    type_dict = {type_map[i]:i for i in range(len(type_map))}
    for i in range(env_types.size):
        _type = tuple([env_types[i]] + np.sort(env_types[structure[i]]).tolist())
        next_types.append(type_dict.get(_type, -1))
    return np.array(next_types, dtype = int)

"""
def encode_by_env(atom_types: Union[Sequence[int], Sequence[Sequence[int]]], structure: Union[np.ndarray, Sequence[np.ndarray]], 
                  data: Union[Sequence[float], Sequence[Sequence[float]]], max_depth: int = 4) -> EnvState:
    depth = 0
    if isinstance(atom_types[0], int):
        env_types = np.array(atom_types, dtype = int)
    else:
        env_types = np.array(sum([list(atom_t) for atom_t in atom_types], start = []), dtype = int)
    if not isinstance(structure, np.ndarray):
        structure = [structure]
    data = np.array(data, dtype = float)
    num = env_types.size
    type_map = sorted(set(atom_types))

    env_data_map = []
    stored_mask = np.array([False] * num)
    env_type_map = []
    while depth < max_depth:
        ntyp = len(type_map)
        datai = [np.nan] * ntyp
        conflict_mask = np.array([False] * ntyp)
        for t, d in zip(env_types[~stored_mask], data[~stored_mask]):
            if not conflict_mask[t]:
                if np.isnan(datai[t]):
                    datai[t] = d
                elif d != datai[t]:
                    datai[t] = np.nan
                    conflict_mask[t] = True
        env_data_map.append(datai)
        stored_mask |= ~conflict_mask[env_types]
        if depth > 0:
            env_type_map.append(type_map)
        depth += 1
        if np.all(stored_mask):
            break
        else:
            next_env_types, next_type_map = build_next_env(env_types, structure)
            if len(next_type_map) <= len(type_map):
                break
            else:
                env_types = next_env_types
                type_map = next_type_map
    if not np.all(stored_mask):
        raise RuntimeError("Data cannot be stored!")
    env_state = (env_data_map, env_type_map)
    return env_state
"""

def encode_by_env(atom_types: Union[Sequence[int], Sequence[Sequence[int]]], structure: Union[np.ndarray, Sequence[np.ndarray]], 
                  data: Union[Sequence[float], Sequence[Sequence[float]]], max_depth: int = 4) -> EnvState:
    if isinstance(atom_types, np.ndarray):
        env_types = atom_types.flatten()
    elif isinstance(atom_types[0], (int, np.int_)):
        env_types = np.array(atom_types)
    else:
        env_types = np.array(sum([list(atom_t) for atom_t in atom_types], start = []))
    if isinstance(data, np.ndarray):
        data = data.flatten()
    elif isinstance(data[0], (float, np.float_)):
        data = np.array(data)
    else:
        data = np.array(sum([list(d) for d in data], start = []))
    if isinstance(structure, np.ndarray):
        structure = [structure]
    else:
        structure = list(structure)
    data = np.array(data, dtype = float)
    num = env_types.size
    type_map = sorted(set(env_types.tolist()))

    EPS = 1e-5
    env_data_map = []
    stored_mask = np.zeros((num, ), dtype = bool)
    env_type_map = []
    for _ in range(max_depth):
        ntyp = len(type_map)
        env_type_map.append(type_map)
        # Type-Atom: one-hot coding.
        type_atom = np.eye(ntyp, dtype = int)[:, env_types[~stored_mask]]

        active_type = np.any(type_atom.astype(bool), axis = -1)
        type_atom = type_atom[active_type]
        unstored_data = data[None, ~stored_mask]
        data_map = np.take_along_axis(unstored_data, np.argmax(type_atom, axis = -1)[..., None], axis = -1)
        pass_mask = np.all(np.abs((unstored_data - data_map) * type_atom) < EPS, axis = -1)
        active_type[active_type] = pass_mask

        _data = np.empty((ntyp, ), dtype = float)
        _data.fill(np.nan)
        _data[active_type] = data_map[pass_mask, 0]
        env_data_map.append(_data.tolist())
        stored_mask |= active_type[env_types]
        if not np.all(stored_mask):
            env_types, type_map = build_next_env(env_types, structure)
        else:
            env_type_map.pop(0)
            return EnvState(env_data_map, env_type_map)
    else:
        raise RuntimeError("Failed to store the data!")

def decode_by_env(atom_types: Sequence[int], structure: np.ndarray, env_state: EnvState):
    env_types = np.array(atom_types, dtype = int)
    env_data_map, env_type_map = env_state
    max_depth = len(env_data_map)

    data = np.array(env_data_map[0], dtype = float)[env_types]  # depth = 0
    for depth in range(1, max_depth):
        env_types = get_next_env(env_types, env_type_map[depth - 1], structure)
        data_map = np.array(env_data_map[depth] + [np.nan], dtype = float)
        mask = np.isnan(data)
        data[mask] = data_map[env_types[mask]]
        if not np.any(np.isnan(data)):
            return data
    else:
        print(data)
        raise RuntimeError("Failed to infer the data!")

