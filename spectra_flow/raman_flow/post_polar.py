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

class PostPolar(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "polar_setting": BigParameter(dict),
            "wannier_centroid": Artifact(Dict[str, Path]),
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
        polar_setting: Dict[str, Union[str, dict]] = op_in["polar_setting"]
        eps = polar_setting["eps_efield"]
        c_diff = polar_setting["central_diff"]
        wc_dict: Dict[str, np.ndarray] = {}
        for key, p in op_in["wannier_centroid"].items():
            arr = np.loadtxt(p, dtype = float, ndmin = 2)
            wc_dict[key] = arr.reshape(arr.shape[0], -1, 3)

        polar = self.cal_polar(c_diff, eps, wc_dict)
        polar_path = Path("polarizability.raw")
        np.savetxt(polar_path, polar, fmt = "%15.8f")
        return OPIO({
            "polarizability": polar_path
        })
    
    def cal_polar(self, c_diff: bool, eps: float, wc_dict: Dict[str, np.ndarray]) -> np.ndarray:
        '''
        Calculate polarizability from a dictionary of wannier centroids.

        Parameters:
        ----------------------------------
        c_diff: bool. Whether to use central difference.

        eps: float. The magnitude of the electronic field, the unit is a.u.

        wc_dict: Dict[str, np.ndarray]. Key `"ori"` refers to the wc under the zero electronic field. 

        If `c_diff = False`,  
        keys `"x", "y", "z"` refer to the wc array under the electronic field along x/y/z axis respectively.

        If `c_diff = True`,  
        keys `"xp", "xm"` refer to the wc array under the electronic field of positive/negative x respectively. 
        Similarly, keys `"yp", "ym"` and keys `"zp", "zm"` are respective to the y/z axis.

        Each array in wc_dict is the shape of (nframe, natom, 3).

        Return:
        ----------------------------------
        polarizability: np.ndarray. (nframe, natom * 9)
        '''
        v = next(iter(wc_dict.values()))
        polar = np.zeros((v.shape[0], v.shape[1], 3, 3), dtype = float)
        if c_diff:
            polar[:, :, 0, :] = (wc_dict["ef_xp"] - wc_dict["ef_xm"]) / (2 * eps)
            polar[:, :, 1, :] = (wc_dict["ef_yp"] - wc_dict["ef_ym"]) / (2 * eps)
            polar[:, :, 2, :] = (wc_dict["ef_zp"] - wc_dict["ef_zm"]) / (2 * eps)
        else:
            polar[:, :, 0, :] = (wc_dict["ef_x"] - wc_dict["ori"]) / eps
            polar[:, :, 1, :] = (wc_dict["ef_y"] - wc_dict["ori"]) / eps
            polar[:, :, 2, :] = (wc_dict["ef_z"] - wc_dict["ori"]) / eps
        return polar.reshape(polar.shape[0], -1)
