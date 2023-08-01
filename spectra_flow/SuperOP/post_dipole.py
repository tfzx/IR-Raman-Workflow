from typing import Dict
from pathlib import Path
import numpy as np
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
)

class PostDipole(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "confs": Artifact(Path),
            "wannier_centroid": Artifact(Dict[str, Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "labeled_confs": Artifact(Path)
        })

    @OP.exec_sign_check # type: ignore
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        conf_path = op_in["confs"]
        wc = np.loadtxt(op_in["wannier_centroid"]["ori"], dtype = float, ndmin = 2)
        np.save(conf_path / "dipole.npy", wc)
        return OPIO({
            "labeled_confs": conf_path
        })
    