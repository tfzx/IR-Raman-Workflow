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
            "input_setting": BigParameter(dict),
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
        input_setting: Dict[str, Union[str, dict]] = op_in["input_setting"]
        eps = input_setting["polar_params"]["eps_efield"]
        c_diff = input_setting["polar_params"]["central_diff"]
        wc_dict: List[np.ndarray] = []
        for key, p in op_in["wannier_centroid"].items():
            wc_dict[key] = np.loadtxt(p, dtype = float)

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
        polar = np.zeros((wc_dict[0].shape[0], wc_dict[0].shape[1], 9), dtype = float)
        if c_diff:
            polar[:, :, 0:3] = (wc_dict["xp"] - wc_dict["xm"]) / (2 * eps)
            polar[:, :, 3:6] = (wc_dict["yp"] - wc_dict["ym"]) / (2 * eps)
            polar[:, :, 6:9] = (wc_dict["zp"] - wc_dict["zm"]) / (2 * eps)
        else:
            polar[:, :, 0:3] = (wc_dict["x"] - wc_dict["ori"]) / eps
            polar[:, :, 3:6] = (wc_dict["y"] - wc_dict["ori"]) / eps
            polar[:, :, 6:9] = (wc_dict["z"] - wc_dict["ori"]) / eps
        return polar.reshape(polar.shape[0], -1)
