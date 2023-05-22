from math import ceil
from typing import Dict, List, Tuple, Union
from pathlib import Path
import abc
import shutil
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter
)
from dflow.utils import (
    set_directory
)
import dpdata



class PrepPolar(OP, abc.ABC):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "global": BigParameter(dict),
            "input_setting": BigParameter(dict)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "input_setting": BigParameter(dict)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        input_setting: Dict[str, Union[str, dict]] = op_in["input_setting"]
        global_config: dict = op_in["global"]
        input_setting["with_efield"] = True
        eps = global_config["eps_efield"]
        if global_config["central_diff"]:
            input_setting["efields"] = {
                "xp": [eps, 0.0, 0.0],
                "xm": [-eps, 0.0, 0.0],
                "yp": [0.0, eps, 0.0],
                "ym": [0.0, -eps, 0.0],
                "zp": [0.0, 0.0, eps],
                "zm": [0.0, 0.0, -eps]
            }
        else:
            input_setting["efields"] = {
                "x": [eps, 0.0, 0.0],
                "y": [0.0, eps, 0.0],
                "z": [0.0, 0.0, eps]
            }
        return OPIO({
            "input_setting": input_setting
        })
