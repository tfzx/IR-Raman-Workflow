from typing import Sequence
import numpy as np
from collections import Counter, namedtuple

EnvState = namedtuple("EnvState", ["EnvDataMap", "EnvTypeMap"])

def build_next_env(env_types: np.ndarray, structure: np.ndarray):
    env_types = np.array(env_types, dtype = int)
    next_type_dict = {}
    next_type_map = []
    next_types = []
    for i in range(env_types.size):
        neighbours = Counter(env_types[structure[i]])
        _type = tuple([env_types[i]] + sorted(neighbours.elements()))
        if _type not in next_type_dict:
            next_type_map.append(_type)
            next_type_dict[_type] = len(next_type_dict)
        next_types.append(next_type_dict[_type])
    return np.array(next_types, dtype = int), next_type_map

def get_next_env(env_types: np.ndarray, type_map: dict, structure: np.ndarray):
    env_types = np.array(env_types, dtype = int)
    next_types = []
    type_dict = {type_map[i]:i for i in range(len(type_map))}
    for i in range(env_types.size):
        neighbours = Counter(env_types[structure[i]])
        _type = tuple([env_types[i]] + sorted(neighbours.elements()))
        next_types.append(type_dict.get(_type, -1))
    return np.array(next_types, dtype = int)

def encode_by_env(atom_types: Sequence[int], structure: np.ndarray, data: Sequence[float], max_depth: int = 4) -> EnvState:
    depth = 0
    env_types = np.array(atom_types, dtype = int)
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

def encode_by_env2(atom_types: Sequence[int], structure: np.ndarray, data: Sequence[float], max_depth: int = 4) -> EnvState:
    EPS = 1e-5
    depth = 0
    env_types = np.array(atom_types, dtype = int)
    data = np.array(data, dtype = float)
    num = env_types.size
    type_map = sorted(set(atom_types))

    env_data_map = []
    stored_mask = np.zeros((num, ), dtype = bool)
    env_type_map = []
    while depth < max_depth:
        if depth > 0:
            env_type_map.append(type_map)
        ntyp = len(type_map)
        datai = np.ones((ntyp, ), dtype = float) * np.nan
        mask = np.eye(ntyp)[:, env_types[~stored_mask]].astype(bool)
        active_mask = np.any(mask, axis = -1)
        mask = mask[active_mask].astype(int)
        active_data = np.take_along_axis(data[None, ~stored_mask], np.argmax(mask, axis = -1, keepdims = True), axis = -1)
        delta = (data[None, ~stored_mask] - active_data) * mask
        pass_mask = np.all(np.abs(delta) < EPS, axis = -1)
        active_mask[active_mask] = pass_mask
        datai[active_mask] = active_data[pass_mask, 0]
        env_data_map.append(datai.tolist())
        stored_mask |= active_mask[env_types]
        if not np.all(stored_mask):
            next_env_types, next_type_map = build_next_env(env_types, structure)
            if len(next_type_map) > len(type_map):
                env_types = next_env_types
                type_map = next_type_map
            else:
                raise RuntimeError("Failed to store the data!")
        else:
            break
        depth += 1
    if not np.all(stored_mask):
        raise RuntimeError("Failed to store the data!")
    env_state = EnvState(env_data_map, env_type_map)
    return env_state

def decode_by_env(atom_types: Sequence[int], structure: np.ndarray, env_state: EnvState):
    env_types = np.array(atom_types, dtype = int)
    num = env_types.size
    env_data_map, env_type_map = env_state
    data = np.ones((num, ), dtype = float) * np.nan
    for depth in range(len(env_data_map)):
        if depth > 0:
            env_types = get_next_env(env_types, env_type_map[depth - 1], structure)
        data_map = np.array(env_data_map[depth], dtype = float)
        data_map = np.concatenate([data_map, np.array([np.nan])], axis = 0)
        mask = np.isnan(data)
        data[mask] = data_map[env_types[mask]]
        if not np.any(np.isnan(data)):
            break
    if np.any(np.isnan(data)):
        print(data)
        raise RuntimeError("Failed to infer the data!")
    return data

