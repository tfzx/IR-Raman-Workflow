from pathlib import Path
import numpy as np
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    BigParameter,
    OPIOSign, 
)

class CalTotalPolar(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "confs": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "polar": Artifact(Path),
            "cal_dipole_python": Artifact(Path, optional = True),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "total_polar": Artifact(Path)
        })

    @OP.exec_sign_check # type: ignore
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        polar = np.load(op_in["polar"])
        if op_in["cal_dipole_python"]:
            import imp
            cal_dipole_python = imp.load_source("dipole_module", str(op_in["cal_dipole_python"]))
            cal_polar = cal_dipole_python.cal_polar
        else:
            cal_polar = self.cal_polar
        total_polar = cal_polar(polar)
        total_polar_path = Path(f"total_polar.npy")
        np.save(total_polar_path, total_polar)
        return OPIO({
            "total_polar": total_polar_path
        })
    
    def cal_polar(self, polar: np.ndarray) -> np.ndarray:
        polar = polar.reshape(polar.shape[0], -1, 3, 3)
        return -np.sum(polar, axis = 1)
