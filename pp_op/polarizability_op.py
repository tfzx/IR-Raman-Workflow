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

class CalPolar(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "wannier_centroid": Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "polarizability": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        wc: List[np.ndarray] = []
        for p in op_in["wannier_centroid"]:
            wc.append(np.loadtxt(p, dtype = float))

        polar = self.cal_polar(wc, 1e-3)
        polar_path = Path("polarizability.raw")
        np.savetxt(polar_path, polar, fmt = "%15.8f")
        return OPIO({
            "polarizability": polar_path
        })
    
    def cal_polar(self, wc_list: List[np.ndarray], eps: float) -> np.ndarray:
        '''
        Calculate polarizability from a list of wannier centroid.

        Parameters:
        ----------------------------------
        wc_list: List[np.ndarray]. If it contains 4 arrays, wc_list[0] is the wc under the zero electronic field, and wc_list[1:4]
        is the wc array under the electronic field along x/y/z axis respectively.
        If it contains 6 arrays, wc_list[0:2] is the wc array under the electronic field of positive/negative x respectively.
        Similarly, wc[2:4] and wc[4:6] is respective to the y/z axis.
        Each array in wc_list is the shape of (nframe, natom, 3)

        eps: float. The magnitude of the electronic field, the unit is a.u.

        Return:
        ----------------------------------
        polarizability: np.ndarray. (nframe, natom * 9)
        '''
        polar = np.zeros((wc_list[0].shape[0], wc_list[0].shape[1], 9), dtype = float)
        if len(wc_list) == 4:
            polar[:, :, 0:3] = (wc_list[1] - wc_list[0]) / eps
            polar[:, :, 3:6] = (wc_list[2] - wc_list[0]) / eps
            polar[:, :, 6:9] = (wc_list[3] - wc_list[0]) / eps
        elif len(wc_list) == 6:
            polar[:, :, 0:3] = (wc_list[1] - wc_list[0]) / (2 * eps)
            polar[:, :, 3:6] = (wc_list[3] - wc_list[2]) / (2 * eps)
            polar[:, :, 6:9] = (wc_list[5] - wc_list[4]) / (2 * eps)
        else:
            raise RuntimeError("Incompatoble wannier centroid list!")
        return polar.reshape(polar.shape[0], -1)
