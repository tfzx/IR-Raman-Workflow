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
from spectra_flow.utils import recurcive_update



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

    @OP.exec_sign_check # type: ignore
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        mlwf_setting: Dict[str, Union[str, dict]] = op_in["mlwf_setting"]
        polar_setting: dict = op_in["polar_setting"]
        mlwf_setting["with_efield"] = True # type: ignore
        ef_type = polar_setting.get("ef_type", "enthalpy").lower()
        mlwf_setting["ef_type"] = ef_type
        eps = polar_setting["eps_efield"]
        if "ef_params" in polar_setting:
            recurcive_update(mlwf_setting["dft_params"], polar_setting["ef_params"]) # type: ignore
        if "efields" in polar_setting:
            mlwf_setting["efields"] = polar_setting["efields"]
        else:
            if ef_type == "enthalpy":
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
            elif ef_type == "saw":
                if polar_setting["central_diff"]:
                    mlwf_setting["efields"] = {
                        "d1p": [1, eps],
                        "d1m": [1, -eps],
                        "d2p": [2, eps],
                        "d2m": [2, -eps],
                        "d3p": [3, eps],
                        "d3m": [3, -eps],
                    }
                else:
                    mlwf_setting["efields"] = {
                        "d1": [1, eps],
                        "d2": [2, eps],
                        "d3": [3, eps],
                    }
        return OPIO({
            "mlwf_setting": mlwf_setting
        })
