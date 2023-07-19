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
from spectra_flow.utils import read_conf, inv_cells

class PostPolar(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "polar_setting": BigParameter(dict),
            "confs": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "wannier_centroid": Artifact(Dict[str, Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "polarizability": Artifact(Path),
            "labeled_confs": Artifact(Path)
        })

    @OP.exec_sign_check # type: ignore
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        polar_setting: Dict[str, Union[str, dict]] = op_in["polar_setting"]
        eps = polar_setting["eps_efield"]
        c_diff = polar_setting["central_diff"]
        conf_path = op_in["confs"]
        confs = read_conf(conf_path, op_in["conf_fmt"])
        wc_dict: Dict[str, np.ndarray] = {}
        for key, p in op_in["wannier_centroid"].items():
            arr = np.loadtxt(p, dtype = float, ndmin = 2)
            wc_dict[key] = arr.reshape(arr.shape[0], -1, 3)
        ef_type = polar_setting.get("ef_type", "enthalpy").lower() # type: ignore
        polar = self.cal_polar(c_diff, eps, ef_type, confs, wc_dict) # type: ignore
        polar_path = Path("polarizability.raw")
        np.savetxt(polar_path, polar)
        np.save(conf_path / "polarizability.npy", polar)
        return OPIO({
            "polarizability": polar_path,
            "labeled_confs": conf_path
        })
    
    def cal_polar(self, c_diff: bool, eps: float, ef_type: str, confs: dpdata.System, wc_dict: Dict[str, np.ndarray]) -> np.ndarray:
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
            if ef_type == "enthalpy":
                keys = [
                    ["ef_xp", "ef_xm"],
                    ["ef_yp", "ef_ym"],
                    ["ef_zp", "ef_zm"],
                ]
            elif ef_type == "saw":
                keys = [
                    ["ef_d1p", "ef_d1m"],
                    ["ef_d2p", "ef_d2m"],
                    ["ef_d3p", "ef_d3m"],
                ]
            delta = 2 * eps
        else:
            if ef_type == "enthalpy":
                keys = [
                    ["ef_x", "ori"],
                    ["ef_y", "ori"],
                    ["ef_z", "ori"],
                ]
            elif ef_type == "saw":
                keys = [
                    ["ef_d1", "ori"],
                    ["ef_d2", "ori"],
                    ["ef_d3", "ori"],
                ]
            delta = eps
        for dir in range(3):
            try:
                polar[:, :, dir, :] = (wc_dict[keys[dir][0]] - wc_dict[keys[dir][1]]) / delta # type: ignore
            except KeyError:
                print(f"[WARNING] An error occured while calculating the polarizability along dir {dir}!")
                print("[WARNING] Set the polarizability to 0.")
                pass
        if ef_type == "saw":
            cells: np.ndarray = confs["cells"] # type: ignore
            cells = (cells / np.linalg.norm(cells, ord = 2, axis = -1, keepdims = True))
            inv_c = inv_cells(cells)
            cells = cells.transpose(0, -1, -2) * np.linalg.norm(inv_c, ord = 2, axis = -2, keepdims = True)
            polar = np.matmul(cells[..., np.newaxis, :, :], polar)
        return polar.reshape(polar.shape[0], -1)
