from typing import Dict, List, Tuple, Union
from pathlib import Path
import dpdata, numpy as np, json, abc
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter,
)
from dflow.utils import (
    set_directory,
    run_command
)
from spectra_flow.dp.infer import model_eval
from spectra_flow.utils import read_conf

class DpPredict(OP, abc.ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "dp_setting": BigParameter(dict),
            "sampled_system": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "frozen_model": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "predicted_tensor": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        sys_path: Path = op_in["sampled_system"]
        conf_fmt = op_in["conf_fmt"]
        smp_sys = read_conf(sys_path, conf_fmt)
        predicted_tensor = self.eval(op_in["frozen_model"], op_in["dp_setting"], smp_sys)
        tensor_path = Path("predicted_tensor.npy")
        np.save(tensor_path, predicted_tensor)
        return OPIO({
            "predicted_tensor": tensor_path
        })

    @abc.abstractmethod
    def eval(self, frozen_model: Path, dp_setting: dict, smp_sys: dpdata.System) -> np.ndarray:
        pass

class DWannPredict(DpPredict):
    def eval(self, frozen_model, dp_setting: dict, smp_sys: dpdata.System) -> np.ndarray:
        from deepmd.infer import DeepDipole
        deep_wannier = DeepDipole(frozen_model)
        predicted_wc = model_eval(deep_wannier, smp_sys)
        if "amplif" in dp_setting:
            predicted_wc /= dp_setting["amplif"]
        return predicted_wc

class DPolarPredict(DpPredict):
    def eval(self, frozen_model, dp_setting: dict, smp_sys: dpdata.System) -> np.ndarray:
        from deepmd.infer import DeepPolar
        deep_polar = DeepPolar(frozen_model)
        predicted_polar = model_eval(deep_polar, smp_sys)
        return predicted_polar
