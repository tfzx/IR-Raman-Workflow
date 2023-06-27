from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from spectra_flow.utils import complete_by_default

class MLWFReaderQeW90:
    """
        This class is a wrapper to read the mlwf_setting of qe+wannier90.
    """
    DEFAULT_MLWF = {
        "name": "name",
        "dft_params": {
            "cal_type": "scf+nscf",
            "pw2wan_params": {
                "inputpp": {
                    "spin_component": None,
                    "write_mmn": True,
                    "write_amn": True,
                    "write_unk": False
                }
            },
            "atomic_species": {}
        }
    }
    DEFAULT_QE = {
        "dft_params": {
            "scf_params": {
                "control": {
                    "restart_mode"  : "from_scratch",
                    "prefix"        : "name",
                    "outdir"        : "out",
                    "pseudo_dir"    : "../../pseudo",
                },
            },
            "nscf_params": {},
        }
    }
    DEFAULT_W90 = {
        "wan_params": {
            "dis_num_iter": 400,
            "num_iter": 100,
            "write_xyz": True,
            "translate_home_cell": True,
            "guiding_centres": True
        }
    }
    def __init__(self, mlwf_setting: Dict[str, Union[str, dict]], if_copy = True, if_print = False) -> None:
        self.mlwf_setting = deepcopy(mlwf_setting) if if_copy else mlwf_setting
        self.name = self.mlwf_setting.get("name", "name")
        self._default()
        self.run_nscf: bool = self.dft_params["cal_type"] == "scf+nscf"
        if if_print:
            import json
            print(json.dumps(self.mlwf_setting, indent=4))
        # if "num_wann" in mlwf_setting["wannier90_params"]["wan_params"]:
        #     mlwf_setting["num_wann"] = mlwf_setting["wannier90_params"]["wan_params"]["num_wann"]
        pass

    def _default(self):
        self.DEFAULT_QE["dft_params"]["scf_params"]["control"]["prefix"] = self.name
        complete_by_default(self.mlwf_setting, self.DEFAULT_MLWF)
        if "qe_params" in self.dft_params and "scf_params" not in self.dft_params:
            self.dft_params["scf_params"] = self.dft_params["qe_params"]
        complete_by_default(self.mlwf_setting, self.DEFAULT_QE)
        self.scf_params["control"]["pseudo_dir"] = "../../pseudo"
        if self.multi_w90_params is None:
            complete_by_default(
                self.mlwf_setting, {
                    "wannier90_params": self.DEFAULT_W90
                }
            )
        else:
            for key in self.multi_w90_params:
                complete_by_default(
                    self.multi_w90_params[key], self.DEFAULT_W90
                )

    @property
    def qe_iter(self):
        qe_keys = ["ori"]
        if self.with_efield:
            qe_keys += [f"ef_{name}" for name in self.efields] # type: ignore
        return qe_keys

    @property
    def dft_params(self) -> dict:
        return self.mlwf_setting["dft_params"] # type: ignore

    @property
    def scf_params(self) -> Dict[str, Any]:
        if "scf_params" in self.dft_params:
            return self.dft_params["scf_params"]
        else:
            return self.dft_params["qe_params"] # TODO: will be deprecated

    @property
    def nscf_params(self) -> Optional[Dict[str, Any]]:
        return self.dft_params.get("nscf_params", None)

    @property
    def pw2wan_params(self) -> dict:
        return self.dft_params["pw2wan_params"]

    @property
    def w90_params(self) -> Optional[Dict[str, dict]]:
        return self.mlwf_setting.get("wannier90_params", None) # type: ignore

    @property
    def multi_w90_params(self) -> Optional[Dict[str, Dict[str, dict]]]:
        return self.mlwf_setting.get("multi_w90_params", None) # type: ignore

    @property
    def atomic_species(self) -> Optional[Dict[str, Dict[str, Union[str, float]]]]:
        return self.dft_params.get("atomic_species", None)

    @property
    def with_efield(self) -> bool:
        return self.mlwf_setting.get("with_efield", False) and (self.efields is not None) # type: ignore
    
    @property
    def efields(self) -> Optional[Dict[str, List[Union[int, float]]]]:
        return self.mlwf_setting.get("efields", None) # type: ignore

    @property
    def ef_type(self) -> str:
        return self.mlwf_setting.get("ef_type", "enthalpy") # type: ignore

    def seed_name(self, qe_key: str, w90_key: str):
        return f"{self.name}_{qe_key}_{w90_key}"

    def scf_name(self, qe_key: str):
        return f"scf_{qe_key}.in"

    def nscf_name(self, qe_key: str):
        return f"nscf_{qe_key}.in"

    def get_w90_params_dict(self):
        multi_w90_params = self.multi_w90_params
        if multi_w90_params is None:
            assert self.w90_params is not None
            w90_params_dict = {"": self.w90_params}
        else:
            w90_params_dict = multi_w90_params
        return w90_params_dict
    
    def get_kgrid(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        _scf_grid = None; _nscf_grid = None
        dft_params = self.dft_params
        _scf_grid = dft_params.get("k_grid", None)
        if "kmesh" in self.mlwf_setting:
            if "scf_grid" in self.mlwf_setting["kmesh"]:
                _scf_grid = self.mlwf_setting["kmesh"]["scf_grid"] # type: ignore
            if "nscf_grid" in self.mlwf_setting["kmesh"]:
                _nscf_grid = self.mlwf_setting["kmesh"]["nscf_grid"] # type: ignore
        scf_grid: Optional[Tuple[int, int, int]] = None
        if _scf_grid is not None:
            scf_grid = tuple(_scf_grid) # type: ignore
        assert scf_grid is not None
        nscf_grid: Tuple[int, int, int] = scf_grid
        if _nscf_grid is not None:
            nscf_grid = tuple(_nscf_grid) # type: ignore
        return scf_grid, nscf_grid

    def get_qe_params_dict(self):
        scf_params = self.scf_params
        qe_params_dict: Dict[str, Dict[str, Any]] = {} # TODO: write into mlwf_setting
        if self.with_efield:
            ef_type = self.ef_type
            qe_params_dict["ori"] = self.complete_ef(scf_params, efield = None, ef_type = ef_type, is_ori = True)
            for ef_name, efield in self.efields.items(): # type: ignore
                qe_params_dict[f"ef_{ef_name}"] = self.complete_ef(scf_params, efield, ef_type, is_ori = False)
        else:
            qe_params_dict["ori"] = scf_params
        return qe_params_dict

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
