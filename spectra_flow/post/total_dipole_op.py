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
from spectra_flow.post.cal_wannier_centroid import cal_wc_h2o
from spectra_flow.post.cal_dipole import calculate_dipole_h2o

class CalTotalDipole(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "confs": Artifact(Path),
            "wannier_centroid": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "total_dipole": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        confs = dpdata.System(op_in["confs"], fmt='deepmd/npy', type_map = ['O', 'H'])
        wc = np.load(op_in["wannier_centroid"])
        dipole = self.cal_dipole(confs, wc)
        dipole_path = Path(f"total_dipole.npy")
        np.save(dipole_path, dipole)
        return OPIO({
            "total_dipole": dipole_path
        })
    
    def cal_dipole(self, confs: dpdata.System, wc: np.ndarray) -> np.ndarray:
        mask_O = confs["atom_types"] == 0
        box = np.diagonal(confs["cells"], axis1 = -2, axis2 = -1)
        coords = confs["coords"] % box.reshape(-1, 1, 3)
        nframes = coords.shape[0]
        wc = wc.reshape(nframes, -1, 3)
        return calculate_dipole_h2o(
            coords[:, mask_O], coords[:, ~mask_O], box, wc
        ).reshape(confs.get_nframes(), -1)
