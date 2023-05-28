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
            "polar_setting": BigParameter(dict),
            "mlwf_setting": BigParameter(dict)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "mlwf_setting": BigParameter(dict)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        mlwf_setting: Dict[str, Union[str, dict]] = op_in["mlwf_setting"]
        polar_setting: dict = op_in["polar_setting"]
        mlwf_setting["with_efield"] = True
        mlwf_setting["ef_type"] = polar_setting.get("ef_type", "enthalpy")
        eps = polar_setting["eps_efield"]
        if polar_setting["central_diff"]:
            mlwf_setting["efields"] = {
                "xp": [eps, 0.0, 0.0],
                "xm": [-eps, 0.0, 0.0],
                "yp": [0.0, eps, 0.0],
                "ym": [0.0, -eps, 0.0],
                "zp": [0.0, 0.0, eps],
                "zm": [0.0, 0.0, -eps]
            }
        else:
            mlwf_setting["efields"] = {
                "x": [eps, 0.0, 0.0],
                "y": [0.0, eps, 0.0],
                "z": [0.0, 0.0, eps]
            }
        return OPIO({
            "mlwf_setting": mlwf_setting
        })
