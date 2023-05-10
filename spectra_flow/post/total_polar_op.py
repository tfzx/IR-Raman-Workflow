from pathlib import Path
import numpy as np
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
)

class CalTotalPolar(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "polar": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "total_polar": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        polar = np.load(op_in["polar"])
        total_polar = self.cal_total_polar(polar)
        total_polar_path = Path(f"total_polar.npy")
        np.save(total_polar_path, total_polar)
        return OPIO({
            "total_polar": total_polar_path
        })
    
    def cal_total_polar(self, polar: np.ndarray) -> np.ndarray:
        polar = polar.reshape(polar.shape[0], -1, 3, 3)
        return np.sum(polar, axis = 1)
