from typing import Dict, List, Optional, Union
from pathlib import Path
import dpdata, numpy as np, shutil
from spectra_flow.mlwf.prepare_input_op import Prepare
from spectra_flow.mlwf.run_mlwf_op import RunMLWF
from spectra_flow.mlwf.collect_wfc_op import CollectWFC
from spectra_flow.mlwf.inputs import QeParamsConfs, QeParams, Wannier90Inputs, complete_qe, complete_wannier90, complete_pw2wan
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

    def get_writers(self, input_setting: Dict[str, Union[str, dict]], confs: dpdata.System):
        # k_grid must be (1, 1, 1)
        k_grid = input_setting["dft_params"]["k_grid"]
        qe_params = input_setting["dft_params"]["qe_params"]
        efields: Dict[str, List[float]] = input_setting["efields"]
        input_pw2wan = complete_pw2wan(
            input_setting["dft_params"]["pw2wan_params"], 
            f"{self.name}_ori", 
            qe_params["control"]["prefix"],
            qe_params["control"]["outdir"]
        )

        # original MLWF, with lelfield = .false.
        params_ori = self.complete_ef(qe_params, is_ori = True, efield = None)
        input_ori, kpoints_ori = complete_qe(params_ori, "scf", k_grid, confs)
        self.scf_writers = {
            "ori": QeParamsConfs(input_ori, kpoints_ori, input_setting["dft_params"]["atomic_species"], confs)
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
                inputs, kpoints, input_setting["dft_params"]["atomic_species"], confs
            )
            input_pw2wan["inputpp"]["seedname"] = f"{self.name}_{ef_name}"
            self.pw2wan_writers[ef_name] = QeParams(input_pw2wan)

    def init_inputs(self, input_setting: Dict[str, Union[str, dict]], confs: dpdata.System):
        self.name = input_setting["name"]
        assert input_setting["with_efield"]
        complete_by_default(input_setting["dft_params"]["qe_params"], params_default = self.DEFAULT_PARAMS)
        if "num_wann" in input_setting["wannier90_params"]["wan_params"]:
            input_setting["num_wann"] = input_setting["wannier90_params"]["wan_params"]["num_wann"]

        self.get_writers(input_setting, confs)
        
        wan_params, proj, kpoints = complete_wannier90(
            input_setting["wannier90_params"]["wan_params"], 
            input_setting["wannier90_params"]["projections"],
            input_setting["dft_params"]["k_grid"]
        )
        self.wannier90_writer = Wannier90Inputs(wan_params, proj, kpoints, confs)
        return input_setting

    def write_one_frame(self, frame: int):
        for ef_name in self.scf_writers:
            with set_directory(ef_name, mkdir = True):
                Path(f"scf_{ef_name}.in").write_text(self.scf_writers[ef_name].write(frame))
                Path(f"{self.name}_{ef_name}.pw2wan").write_text(self.pw2wan_writers[ef_name].write(frame))
                Path(f"{self.name}_{ef_name}.win").write_text(self.wannier90_writer.write(frame))
        return super().write_one_frame(frame)


class RunEfQeWann(RunMLWF):
    def __init__(self) -> None:
        super().__init__()

    def init_cmd(self, commands: Dict[str, str]):
        self.pw_cmd = commands.get("pw", "pw.x")
        self.pw2wan_cmd = commands.get("pw2wannier", "pw2wannier90.x")
        self.wannier_cmd = commands.get("wannier90", "wannier90.x")
        self.wannier90_pp_cmd = commands.get("wannier90_pp", "wannier90.x")
    
    def run_one_frame(self) -> Path:
        out_dir = Path(self.input_setting["dft_params"]["qe_params"]["control"]["outdir"])
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
        for ef_name in self.input_setting["efields"]:
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


class CollectEfWann(CollectWFC):
    def collect_wfc(self, input_setting: dict, backward: List[Path]) -> Dict[str, Path]:
        if not backward:
            return np.array([])
        self.init_params(input_setting, backward)
        namelist = ["ori"] + [f"ef_{key}" for key in input_setting["efields"].keys()]
        wfc: Dict[str, np.ndarray] = {}
        for key in namelist:
            wfc[key] = np.zeros((len(backward), self.num_wann * 3), dtype = float)
        for frame, p in enumerate(backward):
            with set_directory(p):
                for key in namelist:
                    wfc[key][frame] = self.get_one_frame(f"{self.name}_{key}").flatten()
        wfc_path = {}
        for key in namelist:
            wfc_path[key] = Path(f"wfc_{key}.raw")
            np.savetxt(wfc_path[key], wfc[key], fmt = "%15.8f")
        return wfc_path

    def get_one_frame(self, name: str) -> np.ndarray:
        return np.loadtxt(f'{name}_centres.xyz', dtype = float, skiprows = 2, usecols = [1, 2, 3], max_rows = self.num_wann)
    
    def init_params(self, input_setting: dict, backward: List[Path]):
        self.name = input_setting["name"]
        return super().init_params(input_setting, backward)

    def get_num_wann(self, file_path: Path) -> int:
        num_wann = int(np.loadtxt(file_path / f'{self.name}_ori_centres.xyz', dtype = int, max_rows = 1)) - self.confs.get_natoms()
        return num_wann