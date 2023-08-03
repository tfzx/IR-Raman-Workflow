
from typing import Dict, List, Tuple
from spectra_flow.utils import (
    complete_by_default,
    load_json
)

_default_par = {
    "config": {
        "global": {
            "name": "system",
            "calculation": "ir",
            "dt": 0.0003,
            "nstep": 10000,
            "window": 1000,
            "temperature": 300,
            "width": 240
        },
    },
    "uploads": {
        "frozen_model": {},
        "system": {},
        "other": {}
    }
}

def _default(parameters: dict):
    complete_by_default(parameters, _default_par)

def read_par(parameters: Dict[str, dict]):
    _default(parameters)
    config = parameters["config"]
    global_config = config["global"]
    type_map = global_config["type_map"]
    uploads = parameters["uploads"]
    frozen_model = uploads["frozen_model"]
    system = uploads["system"]
    other = uploads["other"]
    inputs = {}

    read_list = [
        ("global_config", config, ["global"]),
        ("polar_setting", config, ["polar"]),
        ("mlwf_setting", config, ["dipole", "mlwf_setting"]),
        ("task_setting", config, ["dipole", "task_setting"]),
        ("dwann_setting", config, ["deep_wannier"]),  # TODO
        ("dpolar_setting", config, ["deep_polar"]),  # TODO
        ("dp_model", frozen_model, ["deep_potential"]),
        ("dwann_model", frozen_model, ["deep_wannier"]),
        ("dpolar_model", frozen_model, ["deep_polar"]),
        ("pseudo", other, ["pseudo"]),
        ("train_label", other, ["train_label"]),
        ("total_dipole", other, ["total_dipole"]),
        ("total_polar", other, ["total_polar"]),
        ("cal_dipole_python", other, ["cal_dipole_python"]),
    ]
    for read_config in read_list:
        read_inputs(inputs, *read_config)

    file_config_list = [
        ("dwann_setting", "train_inputs"),
        ("dpolar_setting", "train_inputs"),
        ("mlwf_setting", ),
        ("polar_setting", ),
        ("task_setting", )
    ]
    for keys in file_config_list:
        config_from_file(inputs, keys)
    sys_fmt_map = {
        "train_confs": "train_conf_fmt",
        "labeled_sys": "labeled_sys_fmt",
        "sampled_system": "sys_fmt",
        "init_conf": "init_conf_fmt",
    }
    for sys_name, fmt_name in sys_fmt_map.items():
        sys_path, sys_fmt = load_system(type_map, system, sys_name)
        if sys_path is not None:
            inputs[sys_name] = sys_path
            inputs[fmt_name] = sys_fmt
    return inputs

def config_from_file(inputs: dict, keys: Tuple[str]):
    for key in keys[:-1]:
        if not key in inputs:
            return
        else:
            inputs = inputs[key]
    last_key = keys[-1]
    if not last_key in inputs:
        return
    if isinstance(inputs[last_key], str):
        inputs[last_key] = load_json(inputs[last_key])
    elif isinstance(inputs[last_key], list):
        config_list = []
        for p in inputs[last_key]:
            config_list.append(load_json(p))
        inputs[last_key] = config_list


def read_inputs(inputs_dict: dict, name: str, par: dict, keys: Tuple[str]):
    if keys[0] in par:
        for key in keys:
            par = par[key]
        inputs_dict[name] = par

def load_system(type_map, up_sys: dict, name: str):
    if name not in up_sys:
        return None, None
    sys = up_sys[name]
    sys_path = sys["path"]
    sys_fmt = {
        "type_map": type_map
    }
    if "fmt" in sys:
        sys_fmt["fmt"] = sys["fmt"]
    return sys_path, sys_fmt
    