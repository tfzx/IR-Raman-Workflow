from pathlib import Path
import dpdata, numpy as np, abc, importlib
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter,
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
            "sys_fmt": BigParameter(dict),
            "frozen_model": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "predicted_tensor": Artifact(Path)
        })

    @OP.exec_sign_check # type: ignore
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        sys_path: Path = op_in["sampled_system"]
        sys_fmt = op_in["sys_fmt"]
        smp_sys = read_conf(sys_path, sys_fmt)
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
        DeepDipole = importlib.import_module("deepmd.infer").DeepDipole
        deep_wannier = DeepDipole(frozen_model)
        predict_batch = dp_setting.get("predict_batch", 128)
        predicted_wc = model_eval(deep_wannier, smp_sys, set_size = predict_batch)
        predicted_wc /= dp_setting.get("amplif", 1.0)
        return predicted_wc

class DPolarPredict(DpPredict):
    def eval(self, frozen_model, dp_setting: dict, smp_sys: dpdata.System) -> np.ndarray:
        DeepPolar = importlib.import_module("deepmd.infer").DeepPolar
        deep_polar = DeepPolar(frozen_model)
        predict_batch = dp_setting.get("predict_batch", 128)
        predicted_polar = model_eval(deep_polar, smp_sys, set_size = predict_batch)
        predicted_polar /= dp_setting.get("amplif", 1.0)
        return predicted_polar
