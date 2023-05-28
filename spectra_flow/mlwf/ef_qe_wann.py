from typing import Dict, List, Optional, Union
from types import ModuleType
from pathlib import Path
import dpdata, numpy as np, shutil
from spectra_flow.mlwf.mlwf_ops import Prepare, RunMLWF, CollectWFC
from spectra_flow.mlwf.qe_wannier90 import CollectWann
from spectra_flow.mlwf.inputs import (
    QeParamsConfs, 
    QeParams, 
    Wannier90Inputs, 
    complete_qe, 
    complete_wannier90, 
    complete_pw2wan
)
from spectra_flow.utils import complete_by_default
from copy import deepcopy
from dflow.utils import set_directory

class PrepareEfQeWann(Prepare):
    DEFAULT_PARAMS = {
        "control": {
            "prefix"        : "h2o",
            "outdir"        : "out",
            "pseudo_dir"    : "../../pseudo",
        }
    }

    def __init__(self):
        super().__init__()

    def complete_ef(self, qe_params: Dict[str, dict], is_ori: bool, efield: Optional[List[float]]):
        params = deepcopy(qe_params)
        params["control"].update({
            "restart_mode": "from_scratch" if is_ori else "restart",
            "lelfield": not is_ori
        })
        if not efield:
            efield = [0.0, 0.0, 0.0]
        params["electrons"].update({
            "efield_cart(1)": efield[0],
            "efield_cart(2)": efield[1],
            "efield_cart(3)": efield[2]
        })
        return params

    def get_writers(self, mlwf_setting: Dict[str, Union[str, dict]], confs: dpdata.System):
        # k_grid must be (1, 1, 1)
        k_grid = mlwf_setting["dft_params"]["k_grid"]
        qe_params = mlwf_setting["dft_params"]["qe_params"]
        efields: Dict[str, List[float]] = mlwf_setting["efields"]
        input_pw2wan = complete_pw2wan(
            mlwf_setting["dft_params"]["pw2wan_params"], 
            f"{self.name}_ori", 
            qe_params["control"]["prefix"],
            qe_params["control"]["outdir"]
        )

        # original MLWF, with lelfield = .false.
        params_ori = self.complete_ef(qe_params, is_ori = True, efield = None)
        input_ori, kpoints_ori = complete_qe(params_ori, "scf", k_grid, confs)
        self.scf_writers = {
            "ori": QeParamsConfs(input_ori, kpoints_ori, mlwf_setting["dft_params"]["atomic_species"], confs)
        }
        self.pw2wan_writers = {
            "ori": QeParams(input_pw2wan)
        }

        # lelfield = .true., efield in efields.
        # See PrepareEfQeWann.complete_ef
        for ef_name, efield in efields.items():
            ef_name = f"ef_{ef_name}"
            params = self.complete_ef(qe_params, is_ori = False, efield = efield)
            inputs, kpoints = complete_qe(params, "scf", k_grid, confs)
            self.scf_writers[ef_name] = QeParamsConfs(
                inputs, kpoints, mlwf_setting["dft_params"]["atomic_species"], confs
            )
            input_pw2wan["inputpp"]["seedname"] = f"{self.name}_{ef_name}"
            self.pw2wan_writers[ef_name] = QeParams(input_pw2wan)

    def init_inputs(self, 
                    mlwf_setting: Dict[str, Union[str, dict]], 
                    confs: dpdata.System,
                    wc_python: ModuleType = None) -> Dict[str, Union[str, dict]]:
        self.name = mlwf_setting["name"]
        assert mlwf_setting["with_efield"]
        complete_by_default(mlwf_setting["dft_params"]["qe_params"], params_default = self.DEFAULT_PARAMS)
        if "num_wann" in mlwf_setting["wannier90_params"]["wan_params"]:
            mlwf_setting["num_wann"] = mlwf_setting["wannier90_params"]["wan_params"]["num_wann"]

        self.get_writers(mlwf_setting, confs)
        
        wan_params, proj, kpoints = complete_wannier90(
            mlwf_setting["wannier90_params"]["wan_params"], 
            mlwf_setting["wannier90_params"]["projections"],
            mlwf_setting["dft_params"]["k_grid"]
        )
        self.wannier90_writer = Wannier90Inputs(wan_params, proj, kpoints, confs)
        return mlwf_setting

    def prep_one_frame(self, frame: int):
        for ef_name in self.scf_writers:
            with set_directory(ef_name, mkdir = True):
                Path(f"scf_{ef_name}.in").write_text(self.scf_writers[ef_name].write(frame))
                Path(f"{self.name}_{ef_name}.pw2wan").write_text(self.pw2wan_writers[ef_name].write(frame))
                Path(f"{self.name}_{ef_name}.win").write_text(self.wannier90_writer.write(frame))


class RunEfQeWann(RunMLWF):
    def __init__(self) -> None:
        super().__init__()

    def init_cmd(self, commands: Dict[str, str]):
        self.pw_cmd = commands.get("pw", "pw.x")
        self.pw2wan_cmd = commands.get("pw2wannier", "pw2wannier90.x")
        self.wannier_cmd = commands.get("wannier90", "wannier90.x")
        self.wannier90_pp_cmd = commands.get("wannier90_pp", "wannier90.x")
    
    def run_one_frame(self) -> Path:
        out_dir = Path(self.mlwf_setting["dft_params"]["qe_params"]["control"]["outdir"])
        ori_p = Path("ori")
        backward_dir = Path(self.backward_dir_name)
        backward_dir.mkdir()
        back_abs = backward_dir.absolute()
        with set_directory(ori_p):
            self.run(" ".join([self.pw_cmd, "-input", "scf_ori.in"]))
            self.run(" ".join([self.wannier90_pp_cmd, "-pp", f"{self.name}_ori"]))
            self.run(" ".join([self.pw2wan_cmd]), input=Path(f"{self.name}_ori.pw2wan").read_text())
            self.run(" ".join([self.wannier_cmd, f"{self.name}_ori"]))
            shutil.copy(f"{self.name}_ori_centres.xyz", back_abs)
        ori_p = ori_p.absolute()
        for ef_name in self.mlwf_setting["efields"]:
            ef_name = f"ef_{ef_name}"
            with set_directory(ef_name):
                shutil.copytree(ori_p / out_dir, out_dir)
                self.run(" ".join([
                    self.pw_cmd, "-input", f"scf_{ef_name}.in"
                ]))
                self.run(" ".join([
                    self.wannier90_pp_cmd, "-pp", f"{self.name}_{ef_name}"
                ]))
                self.run(" ".join([self.pw2wan_cmd]), 
                    input=Path(f"{self.name}_{ef_name}.pw2wan").read_text()
                )
                self.run(" ".join([self.wannier_cmd, f"{self.name}_{ef_name}"]))
                shutil.rmtree(out_dir)
                shutil.copy(f"{self.name}_{ef_name}_centres.xyz", back_abs)
        shutil.rmtree(ori_p / out_dir)
        
        return backward_dir


class CollectEfWann(CollectWann):
    def init_name_dict(self, mlwf_setting: dict):
        name = mlwf_setting["name"]
        keylist = ["ori"] + [f"ef_{key}" for key in mlwf_setting["efields"].keys()]
        self.name_dict = {
            key: f"{name}_{key}" for key in keylist
        }