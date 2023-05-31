from typing import Dict, List, Optional, Tuple, Union
from copy import deepcopy
from spectra_flow.utils import complete_by_default

class MLWFReaderQeW90:
    """
        This class is a tool to read the mlwf_setting which uses qe+wannier90.
    """
    DEFAULT_MLWF = {
        "name": "name",
        "dft_params": {
            "cal_type": "scf+nscf",
            "qe_params": {
                "control": {
                    "restart_mode"  : "from_scratch",
                    "prefix"        : "h2o",
                    "outdir"        : "out",
                    "pseudo_dir"    : "../../pseudo",
                },
            },
            "pw2wan_params": {
                "inputpp": {
                    "spin_component": None,
                    "write_mmn": True,
                    "write_amn": True,
                    "write_unk": False
                }
            }
        },
        "wannier90_params": {
            "wan_params": {
                "dis_num_iter": 400,
                "num_iter": 100,
                "write_xyz": True,
                "translate_home_cell": True,
                "guiding_centres": True
            }
        }
    }
    def __init__(self, mlwf_setting: Dict[str, Union[str, dict]], if_copy = True) -> None:
        self.mlwf_setting = deepcopy(mlwf_setting) if if_copy else mlwf_setting
        self.name = self.mlwf_setting.get("name", "name")
        self.run_nscf: bool = self.dft_params["cal_type"] == "scf+nscf"
        # if "num_wann" in mlwf_setting["wannier90_params"]["wan_params"]:
        #     mlwf_setting["num_wann"] = mlwf_setting["wannier90_params"]["wan_params"]["num_wann"]
        pass

    def default(self):
        self.DEFAULT_MLWF["dft_params"]["qe_params"]["control"]["prefix"] = self.name
        complete_by_default(self.mlwf_setting, self.DEFAULT_MLWF)
        self.scf_params["control"]["pseudo_dir"] = "../../pseudo"

    @property
    def qe_iter(self):
        qe_keys = ["ori"]
        if self.with_efield:
            qe_keys += [f"ef_{name}" for name in self.efields]
        return qe_keys

    @property
    def dft_params(self) -> dict:
        return self.mlwf_setting["dft_params"]

    @property
    def scf_params(self) -> Dict[str, Dict[str, Union[str, float, bool]]]:
        if "scf_params" in self.dft_params:
            return self.dft_params["scf_params"]
        else:
            return self.dft_params["qe_params"]

    @property
    def nscf_params(self) -> Dict[str, Dict[str, Union[str, float, bool]]]:
        return self.dft_params.get("nscf_params", None)

    @property
    def pw2wan_params(self):
        return self.dft_params["pw2wan_params"]

    @property
    def w90_params(self) -> Dict[str, dict]:
        return self.mlwf_setting.get("wannier90_params", None)

    @property
    def multi_w90_params(self) -> Dict[str, Dict[str, dict]]:
        return self.mlwf_setting.get("multi_w90_params", None)

    @property
    def atomic_species(self) -> Dict[str, Dict[str, Union[str, float]]]:
        return self.dft_params["atomic_species"]

    @property
    def efields(self) -> Dict[str, List[Union[int, float]]]:
        return self.mlwf_setting.get("efields", None)

    @property
    def with_efield(self):
        return self.mlwf_setting.get("with_efield", False) and self.efields

    @property
    def ef_type(self) -> str:
        return self.mlwf_setting.get("ef_type", "enthalpy")

    def seed_name(self, qe_key: str, w90_key: str):
        return f"{self.name}_{qe_key}_{w90_key}"

    def scf_name(self, qe_key: str):
        return f"scf_{qe_key}.in"

    def nscf_name(self, qe_key: str):
        return f"nscf_{qe_key}.in"

    def get_w90_params_dict(self):
        multi_w90_params = self.multi_w90_params
        if multi_w90_params is None:
            multi_w90_params = {"": self.w90_params}
        return multi_w90_params
    
    def get_kgrid(self) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        scf_grid = None; nscf_grid = None
        dft_params = self.dft_params
        if "k_grid" in dft_params:
            scf_grid = dft_params["k_grid"]
        if "kmesh" in self.mlwf_setting:
            if "scf_grid" in self.mlwf_setting["kmesh"]:
                scf_grid = self.mlwf_setting["kmesh"]["scf_grid"]
            if "nscf_grid" in self.mlwf_setting["kmesh"]:
                nscf_grid = self.mlwf_setting["kmesh"]["nscf_grid"]
        if nscf_grid is None:
            nscf_grid = scf_grid
        return scf_grid, nscf_grid

    @classmethod
    def complete_ef(cls, qe_params: Dict[str, dict], efield: Optional[List[float]], ef_type: str = "enthalpy", is_ori: bool = True):
        params = deepcopy(qe_params)
        ef_type = ef_type.lower()
        if ef_type == "enthalpy":
            params["control"]["restart_mode"] = "from_scratch" if is_ori else "restart"
            params["control"]["lelfield"] = not is_ori
            if not efield:
                efield = [0.0, 0.0, 0.0]
            params["electrons"].update({
                "efield_cart(1)": efield[0],
                "efield_cart(2)": efield[1],
                "efield_cart(3)": efield[2]
            })
        elif ef_type == "saw":
            params["control"]["restart_mode"] = "from_scratch"
            params["control"]["tefield"] = not is_ori
            if not efield:
                efield = [3, 0.0]
            edir, eamp = efield[:2]
            params["system"].update({
                "edir": edir,
                "eamp": eamp
            })
        return params

    def get_qe_params_dict(self):
        scf_params = self.scf_params
        qe_params_dict = {}
        if self.with_efield:
            ef_type = self.ef_type
            qe_params_dict["ori"] = self.complete_ef(scf_params, efield = None, ef_type = ef_type, is_ori = True)
            for ef_name, efield in self.efields.items():
                qe_params_dict[f"ef_{ef_name}"] = self.complete_ef(scf_params, efield, ef_type, is_ori = False)
        else:
            qe_params_dict["ori"] = scf_params
        return qe_params_dict
