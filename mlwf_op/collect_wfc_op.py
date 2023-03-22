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

class CollectWFC(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "name": str,
            "confs": Artifact(Path),
            "backward": Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "wannier_function_centers": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        self.name: str = op_in["name"]
        self.confs = dpdata.System(op_in["confs"], fmt='deepmd/raw', type_map = ['O', 'H'])
        backward: List[Path] = op_in["backward"]

        wfc = self.collect_wfc(backward)
        wfc_path = Path("wfc.raw")
        np.savetxt(wfc_path, wfc, fmt = "%15.8f")
        return OPIO({
            "wannier_function_centers": wfc_path
        })
    
    def collect_wfc(self, backward: List[Path]) -> np.ndarray:
        if not backward:
            return np.array([])
        num_wann = int(np.loadtxt(backward[0] / f'{self.name}_centres.xyz', dtype = int, max_rows = 1)) - self.confs.get_natoms()
        wc = np.zeros((len(backward), num_wann * 3), dtype = float)
        for frame, p in enumerate(backward):
            with set_directory(p):
                wc[frame] = self.get_one_frame(frame, num_wann).flatten()
        return wc

    def get_one_frame(self, frame: int, num_wann: int) -> np.ndarray:
        return np.loadtxt(f'{self.name}_centres.xyz', dtype = float, skiprows = 2, usecols = [1, 2, 3], max_rows = num_wann)