from copy import deepcopy
import shutil
from types import ModuleType
from typing import Dict, List, Optional, Union
from pathlib import Path
import dpdata, numpy as np
from dflow.utils import set_directory
from spectra_flow.mlwf.mlwf_ops import Prepare, RunMLWF, CollectWFC
from spectra_flow.mlwf.inputs import (
    QeParamsConfs, 
    QeParams, 
    Wannier90Inputs, 
    complete_qe, 
    complete_wannier90, 
    complete_pw2wan
)
from spectra_flow.utils import complete_by_default, recurcive_update


class PrepareQeWann(Prepare):
    """
        Do MLWF by Qe-Pwscf and Wannier90. 
        ----------
        Support calculation with electric fields.
        Prepare the inputs file for each frame:
            - `scf.in`
            - `nscf.in` (optional. If `cal_type == "scf+nscf"`, this file will be generated.)
            - `{name}.pw2wan`
            - `{name}.win`
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
            }
        },
        "pw2wan_params": {
            "inputpp": {
                "spin_component": None,
                "write_mmn": True,
                "write_amn": True,
                "write_unk": False
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

    def __init__(self):
        super().__init__()

    def complete_ef(self, qe_params: Dict[str, dict], efield: Optional[List[float]], ef_type: str = "enthalpy", is_ori: bool = True):
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

    def get_qe_inputs(
            self, 
            k_grid: List[int],
            scf_params: Dict[str, dict],
            nscf_params: Optional[Dict[str, dict]],
            pw2wan_params: Dict[str, dict], 
            confs: dpdata.System,
        ):
        input_scf, kpoints_scf = complete_qe(scf_params, "scf", k_grid, confs)
        input_nscf = None; kpoints_nscf = None
        if self.run_nscf:
            _nscf_params = deepcopy(scf_params)
            if nscf_params is not None:
                recurcive_update(_nscf_params, nscf_params)
            _nscf_params["control"]["restart_mode"] = "from_scratch"
            input_nscf, kpoints_nscf = complete_qe(_nscf_params, "nscf", k_grid, confs)
        input_pw2wan = complete_pw2wan(
            pw2wan_params, 
            self.name, 
            input_scf["control"]["prefix"],
            input_scf["control"]["outdir"]
            )
        return input_scf, kpoints_scf, input_nscf, kpoints_nscf, input_pw2wan

    def get_w90_rewriter(self, w90_params: Dict[str, dict], wc_python: ModuleType = None):
        rewrite_atoms = None
        rewrite_proj = None
        if wc_python is not None:
            if w90_params.get("rewrite_atoms", False):
                try:
                    rewrite_atoms = wc_python.rewrite_atoms
                except Exception as e:
                    print(f"[WARNING] An error occurred while importing the method 'rewrite_atoms': {e}")
                    print("Use default atom names.")
            if w90_params.get("rewrite_proj", False):
                try:
                    rewrite_proj = wc_python.rewrite_proj
                except Exception as e:
                    print(f"[WARNING] An error occurred while importing the method 'rewrite_proj': {e}")
                    print("Use projections defined in wannier90_params.")
        return rewrite_atoms, rewrite_proj

    def get_w90_inputs(
            self, 
            k_grid: List[int],
            w90_params: Dict[str, dict],
            scf_params: Dict[str, dict],
            nscf_params: Optional[Dict[str, dict]],
        ):
        wan_params = w90_params["wan_params"]
        if self.run_nscf:
            if "system" in nscf_params and "nbnd" in nscf_params["system"]:
                wan_params["num_bands"] = nscf_params["system"]["nbnd"]
        else:
            if "system" in scf_params and "nbnd" in scf_params["system"]:
                wan_params["num_bands"] = scf_params["system"]["nbnd"]
        wan_params, proj, kpoints = complete_wannier90(
            wan_params, 
            w90_params.get("projections", {}),
            k_grid
        )
        return wan_params, proj, kpoints

    def get_writers(self, mlwf_setting: Dict[str, Union[str, dict]], confs: dpdata.System, wc_python: ModuleType):
        dft_params = mlwf_setting["dft_params"]
        w90_params = mlwf_setting["wannier90_params"]
        k_grid = dft_params["k_grid"]
        scf_params = dft_params["qe_params"]
        nscf_params = dft_params.get("nscf_params", None)
        pw2wan_params = dft_params["pw2wan_params"]
        atomic_species = dft_params["atomic_species"]
        params_dict = {}
        efields: Dict[str, List[float]] = mlwf_setting.get("efields", None)
        if mlwf_setting.get("with_efield", False) and efields:
            ef_type = mlwf_setting["ef_type"]
            params_dict["ori"] = self.complete_ef(scf_params, efield = None, ef_type = ef_type, is_ori = True)
            for ef_name, efield in efields.items():
                params_dict[f"ef_{ef_name}"] = self.complete_ef(scf_params, efield, ef_type, is_ori = False)
        else:
            params_dict["ori"] = scf_params
        rewrite_atoms, rewrite_proj = self.get_w90_rewriter(w90_params, wc_python)
        for key, params in params_dict.items():
            (
                input_scf, kpoints_scf, input_nscf, kpoints_nscf, input_pw2wan
            ) = self.get_qe_inputs(
                k_grid, params, nscf_params, pw2wan_params, confs
            )
            self.scf_writers[key] = QeParamsConfs(input_scf, kpoints_scf, atomic_species, confs)
            if self.run_nscf:
                self.nscf_writers[key] = QeParamsConfs(input_nscf, kpoints_nscf, atomic_species, confs)
            input_pw2wan["inputpp"]["seedname"] = f"{self.name}_{key}"
            self.pw2wan_writers[key] = QeParams(input_pw2wan)

            self.wannier90_writers[key] = Wannier90Inputs(
                *self.get_w90_inputs(
                    k_grid, w90_params, input_scf, input_nscf
                ), 
                confs, rewrite_atoms, rewrite_proj
            )

    def init_inputs(self, mlwf_setting: Dict[str, Union[str, dict]], confs: dpdata.System, wc_python: ModuleType):
        self.name = mlwf_setting.get("name", "name")
        self.run_nscf = mlwf_setting["dft_params"]["cal_type"] == "scf+nscf"
        self.DEFAULT_MLWF["dft_params"]["qe_params"]["control"]["prefix"] = self.name
        complete_by_default(mlwf_setting, self.DEFAULT_MLWF)
        mlwf_setting["dft_params"]["qe_params"]["control"]["pseudo_dir"] = "../../pseudo"
        if "num_wann" in mlwf_setting["wannier90_params"]["wan_params"]:
            mlwf_setting["num_wann"] = mlwf_setting["wannier90_params"]["wan_params"]["num_wann"]
        self.scf_writers: Dict[str, QeParamsConfs] = {}
        if self.run_nscf:
            self.nscf_writers: Dict[str, QeParamsConfs] = {}
        self.pw2wan_writers: Dict[str, QeParams] = {}
        self.wannier90_writers: Dict[str, Wannier90Inputs] = {}
        self.get_writers(mlwf_setting, confs, wc_python)
        
        return mlwf_setting

    def prep_one_frame(self, frame: int):
        for key in self.scf_writers:
            with set_directory(key, mkdir = True):
                Path(f"scf_{key}.in").write_text(self.scf_writers[key].write(frame))
                if self.run_nscf:
                    Path(f"nscf_{key}.in").write_text(self.nscf_writers[key].write(frame))
                Path(f"{self.name}_{key}.pw2wan").write_text(self.pw2wan_writers[key].write(frame))
                Path(f"{self.name}_{key}.win").write_text(self.wannier90_writers[key].write(frame))

class RunQeWann(RunMLWF):
    def __init__(self) -> None:
        super().__init__()

    def init_cmd(self, commands: Dict[str, str]):
        self.pw_cmd = commands.get("pw", "pw.x")
        self.pw2wan_cmd = commands.get("pw2wannier", "pw2wannier90.x")
        self.wannier_cmd = commands.get("wannier90", "wannier90.x")
    
    def run_one_frame(self) -> Path:
        out_dir = Path(self.mlwf_setting["dft_params"]["qe_params"]["control"]["outdir"])
        ori_out_dir = Path("ori_out_temp").absolute()
        ori_p = Path("ori")
        backward_dir = Path(self.backward_dir_name)
        backward_dir.mkdir()
        back_abs = backward_dir.absolute()

        efields: Dict[str, List[float]] = self.mlwf_setting.get("efields", None)
        with_ef = self.mlwf_setting.get("with_efield", False) and bool(efields)
        if with_ef:
            ef_type = self.mlwf_setting["ef_type"]

        with set_directory(ori_p):
            self.run(" ".join([self.wannier_cmd, "-pp", f"{self.name}_ori"]))
            self.run(" ".join([self.pw_cmd, "-input", "scf_ori.in"]))
            if with_ef and ef_type == "enthalpy":
                shutil.copytree(out_dir, ori_out_dir)
            if Path("nscf_ori.in").exists():
                self.run(" ".join([self.pw_cmd, "-input", "nscf_ori.in"]))
            self.run(" ".join([self.pw2wan_cmd]), input=Path(f"{self.name}_ori.pw2wan").read_text())
            self.run(" ".join([self.wannier_cmd, f"{self.name}_ori"]))
            shutil.copy(f"{self.name}_ori_centres.xyz", back_abs)
            shutil.rmtree(out_dir)
        if with_ef:
            for ef_name in self.mlwf_setting["efields"]:
                key = f"ef_{ef_name}"
                with set_directory(key):
                    if ef_type == "enthalpy":
                        shutil.copytree(ori_out_dir, out_dir)
                    self.run(" ".join([
                        self.wannier_cmd, "-pp", f"{self.name}_{key}"
                    ]))
                    self.run(" ".join([
                        self.pw_cmd, "-input", f"scf_{key}.in"
                    ]))
                    if Path(f"nscf_{key}.in").exists():
                        self.run(" ".join([self.pw_cmd, "-input", f"nscf_{key}.in"]))
                    self.run(" ".join([self.pw2wan_cmd]), 
                        input=Path(f"{self.name}_{key}.pw2wan").read_text()
                    )
                    self.run(" ".join([self.wannier_cmd, f"{self.name}_{key}"]))
                    shutil.rmtree(out_dir)
                    shutil.copy(f"{self.name}_{key}_centres.xyz", back_abs)
        
        return backward_dir

class CollectWann(CollectWFC):
    def init_params(self, mlwf_setting: dict, conf_sys: dpdata.System, example_file: Path):
        self.init_name_dict(mlwf_setting)
        super().init_params(mlwf_setting, conf_sys, example_file)

    def init_name_dict(self, mlwf_setting: dict):
        name = mlwf_setting["name"]
        efields: Dict[str, List[float]] = mlwf_setting.get("efields", None)
        with_ef = mlwf_setting.get("with_efield", False) and bool(efields)
        keylist = ["ori"]
        if with_ef:
            keylist += [f"ef_{key}" for key in efields.keys()]
        self.name_dict = {
            key: f"{name}_{key}" for key in keylist
        }

    def get_one_frame(self, frame: int) -> Dict[str, np.ndarray]:
        return {key: self.read_wfc(name) for key, name in self.name_dict.items()}
    
    def read_wfc(self, name: str) -> np.ndarray:
        return np.loadtxt(f'{name}_centres.xyz', dtype = float, skiprows = 2, usecols = [1, 2, 3], max_rows = self.num_wann)

    def get_num_wann(self, conf_sys: dpdata.System, example_file: Path) -> int:
        name = next(iter(self.name_dict.values()))
        num_wann = int(np.loadtxt(
            example_file / f'{name}_centres.xyz', 
            dtype = int, max_rows = 1
        )) - conf_sys.get_natoms()
        return num_wann
