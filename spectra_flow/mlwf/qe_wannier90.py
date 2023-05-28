import shutil
from types import ModuleType
from typing import Dict, List, Optional, Union
from pathlib import Path
import dpdata, numpy as np
from spectra_flow.mlwf.mlwf_ops import Prepare, RunMLWF, CollectWFC
from spectra_flow.mlwf.inputs import (
    QeParamsConfs, 
    QeParams, 
    Wannier90Inputs, 
    complete_qe, 
    complete_wannier90, 
    complete_pw2wan
)
from spectra_flow.utils import complete_by_default


class PrepareQeWann(Prepare):
    DEFAULT_PARAMS = {
        "control": {
            "restart_mode"  : "from_scratch",
            "prefix"        : "h2o",
            "outdir"        : "out",
            "pseudo_dir"    : "../pseudo",
        }
    }

    def __init__(self):
        super().__init__()

    def init_inputs(self, input_setting: Dict[str, Union[str, dict]], confs: dpdata.System, wc_python: ModuleType):
        self.name = input_setting["name"]
        self.run_nscf = input_setting["dft_params"]["cal_type"] == "scf+nscf"

        k_grid = input_setting["dft_params"]["k_grid"]
        self.DEFAULT_PARAMS["control"]["prefix"] = self.name
        qe_params = complete_by_default(input_setting["dft_params"]["qe_params"], params_default = self.DEFAULT_PARAMS)
        if "num_wann" in input_setting["wannier90_params"]["wan_params"]:
            input_setting["num_wann"] = input_setting["wannier90_params"]["wan_params"]["num_wann"]

        input_scf, kpoints_scf = complete_qe(qe_params, "scf", k_grid, confs)
        self.scf_writer = QeParamsConfs(input_scf, kpoints_scf, input_setting["dft_params"]["atomic_species"], confs)
        if self.run_nscf:
            input_nscf, kpoints_nscf = complete_qe(qe_params, "nscf", k_grid, confs)
            self.nscf_writer = QeParamsConfs(input_nscf, kpoints_nscf, input_setting["dft_params"]["atomic_species"], confs)
        input_pw2wan = complete_pw2wan(
            input_setting["dft_params"]["pw2wan_params"], 
            self.name, 
            input_scf["control"]["prefix"],
            input_scf["control"]["outdir"]
            )
        self.pw2wan_writer = QeParams(input_pw2wan)

        wannier90_params = input_setting["wannier90_params"]
        wan_params, proj, kpoints = complete_wannier90(
            wannier90_params["wan_params"], 
            wannier90_params.get("projections", {}),
            input_setting["dft_params"]["k_grid"]
        )
        rewrite_atoms = None
        rewrite_proj = None
        if wc_python is not None:
            if wannier90_params.get("rewrite_atoms", False):
                try:
                    rewrite_atoms = wc_python.rewrite_atoms
                except Exception as e:
                    print(f"[WARNING] An error occurred while importing the method 'rewrite_atoms': {e}")
                    print("Use default atom names.")
            if wannier90_params.get("rewrite_proj", False):
                try:
                    rewrite_proj = wc_python.rewrite_proj
                except Exception as e:
                    print(f"[WARNING] An error occurred while importing the method 'rewrite_proj': {e}")
                    print("Use projections defined in wannier90_params.")
        self.wannier90_writer = Wannier90Inputs(wan_params, proj, kpoints, confs, rewrite_atoms, rewrite_proj)
        return input_setting

    def prep_one_frame(self, frame: int):
        Path("scf.in").write_text(self.scf_writer.write(frame))
        if self.run_nscf:
            Path("nscf.in").write_text(self.nscf_writer.write(frame))
        Path(f"{self.name}.pw2wan").write_text(self.pw2wan_writer.write(frame))
        Path(f"{self.name}.win").write_text(self.wannier90_writer.write(frame))


class RunQeWann(RunMLWF):
    def __init__(self) -> None:
        super().__init__()

    def init_cmd(self, commands: Dict[str, str]):
        self.pw_cmd = commands.get("pw", "pw.x")
        self.pw2wan_cmd = commands.get("pw2wannier", "pw2wannier90.x")
        self.wannier_cmd = commands.get("wannier90", "wannier90.x")
        self.wannier90_pp_cmd = commands.get("wannier90_pp", "wannier90.x")
    
    def run_one_frame(self) -> Path:
        self.run(" ".join([self.pw_cmd, "-input", "scf.in"]))
        if Path("nscf.in").exists():
            self.run(" ".join([self.pw_cmd, "-input", "nscf.in"]))
        self.run(" ".join([self.wannier90_pp_cmd, "-pp", self.name]))
        self.run(" ".join([self.pw2wan_cmd]), input=Path(f"{self.name}.pw2wan").read_text())
        self.run(" ".join([self.wannier_cmd, self.name]))
        self.run("rm -rf out")
        backward_dir = Path(self.backward_dir_name)
        backward_dir.mkdir()
        shutil.copy(f"{self.name}_centres.xyz", backward_dir)
        return backward_dir

class CollectWann(CollectWFC):
    def init_params(self, input_setting: dict, conf_sys: dpdata.System, example_file: Path):
        self.init_name_dict(input_setting)
        super().init_params(input_setting, conf_sys, example_file)

    def init_name_dict(self, input_setting: dict):
        self.name_dict = {
            "ori": input_setting["name"]
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
