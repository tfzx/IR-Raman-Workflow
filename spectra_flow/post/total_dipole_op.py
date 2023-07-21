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
from spectra_flow.post.cal_dipole import calculate_dipole_h2o
from spectra_flow.utils import read_conf, do_pbc

class CalTotalDipole(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "confs": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "wannier_centroid": Artifact(Path),
            "cal_dipole_python": Artifact(Path, optional = True),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "total_dipole": Artifact(Path)
        })

    @OP.exec_sign_check # type: ignore
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        conf_fmt: dict = op_in["conf_fmt"]
        confs = read_conf(op_in["confs"], conf_fmt)
        wc = np.load(op_in["wannier_centroid"])
        
        if op_in["cal_dipole_python"]:
            import imp
            cal_dipole_python = imp.load_source("dipole_module", str(op_in["cal_dipole_python"]))
            cal_dipole = cal_dipole_python.cal_dipole
        else:
            cal_dipole = self.cal_dipole
        dipole = cal_dipole(confs, wc)
        dipole_path = Path(f"total_dipole.npy")
        np.save(dipole_path, dipole)
        return OPIO({
            "total_dipole": dipole_path
        })
    
    def cal_dipole(self, confs: dpdata.System, wc: np.ndarray) -> np.ndarray:
        mask_O = confs["atom_types"] == 0
        coords = do_pbc(confs["coords"], confs["cells"]) # type: ignore
        nframes = confs.get_nframes()
        wc = wc.reshape(nframes, -1, 3)
        return calculate_dipole_h2o(
            coords[:, mask_O], coords[:, ~mask_O], confs["cells"], wc, r_bond = 1.2 # type: ignore
        ).reshape(nframes, -1)
