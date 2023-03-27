from typing import Dict, List, Tuple, Union
from pathlib import Path
import dpdata, numpy as np
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter,
)
from dflow.utils import (
    set_directory
)
from pp_op.cal_wannier_centroid import cal_wc_h2o

class CalWC(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "confs": Artifact(Path),
            "wannier_function_centers": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "wannier_centroid": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        confs = dpdata.System(op_in["confs"], fmt='deepmd/raw', type_map = ['O', 'H'])
        wfc = np.loadtxt(op_in["wannier_function_centers"], dtype = float)

        wc = self.cal_wc(confs, wfc)
        wc_path = Path("atomic_dipole.raw")
        np.savetxt(wc_path, wc, fmt = "%15.8f")
        return OPIO({
            "wannier_centroid": wc_path
        })
    
    def cal_wc(self, confs: dpdata.System, wfc: np.ndarray) -> np.ndarray:
        return cal_wc_h2o(
            wfc.reshape(confs.get_nframes(), -1, 3), 
            confs["coords"][:, confs["atom_types"] == 0], 
            np.diagonal(confs["cells"], axis1 = -2, axis2 = -1)
        ).reshape(confs.get_nframes(), -1)
