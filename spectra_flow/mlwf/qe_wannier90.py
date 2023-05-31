import shutil
from types import ModuleType
from typing import Dict, Union
from pathlib import Path
import dpdata, numpy as np
from dflow.utils import set_directory
from spectra_flow.mlwf.mlwf_ops import Prepare, RunMLWF, CollectWFC
from spectra_flow.mlwf.inputs import (
    QeParamsConfs, 
    QeParams, 
    Wannier90Inputs, 
    get_qe_writers,
    get_pw_w90_writers
)
from spectra_flow.mlwf.mlwf_reader import MLWFReaderQeW90

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
    @classmethod
    def get_w90_rewriter(cls, wc_python: ModuleType = None):
        rewrite_atoms = None; rewrite_proj = None
        if wc_python is not None:
            try:
                rewrite_atoms = wc_python.rewrite_atoms
            except:
                pass
            try:
                rewrite_proj = wc_python.rewrite_proj
            except:
                pass
        return rewrite_atoms, rewrite_proj

    def get_writers(self, mlwf: MLWFReaderQeW90, confs: dpdata.System, wc_python: ModuleType):
        w90_params_dict = mlwf.get_w90_params_dict()
        scf_grid, nscf_grid = mlwf.get_kgrid()
        w90_kgrid = nscf_grid if mlwf.run_nscf else scf_grid
        nscf_params = mlwf.nscf_params
        pw2wan_params = mlwf.pw2wan_params
        atomic_species = mlwf.atomic_species
        qe_params_dict = mlwf.get_qe_params_dict()

        rewrite_atoms, rewrite_proj = self.get_w90_rewriter(wc_python)
        for qe_key, scf_params in qe_params_dict.items():
            (
                scf_writer, nscf_writer, input_scf, input_nscf
            ) = get_qe_writers(
                confs, scf_grid, nscf_grid, scf_params, 
                nscf_params, atomic_species, mlwf.run_nscf
            )
            self.scf_writers[qe_key] = scf_writer
            if mlwf.run_nscf:
                self.nscf_writers[qe_key] = nscf_writer
            
            self.pw2wan_writers[qe_key] = {}
            self.wannier90_writers[qe_key] = {}
            for w90_key, w90_params in w90_params_dict.items():
                (
                    self.pw2wan_writers[qe_key][w90_key], 
                    self.wannier90_writers[qe_key][w90_key]
                ) = get_pw_w90_writers(
                    mlwf.seed_name(qe_key, w90_key), 
                    confs, pw2wan_params, w90_params, 
                    w90_kgrid, input_scf, input_nscf,
                    rewrite_atoms, rewrite_proj
                )

    def init_inputs(self, mlwf_setting: Dict[str, Union[str, dict]], confs: dpdata.System, wc_python: ModuleType):
        """
            Each dir runs one qe calculations (scf+[nscf]) and some wannier90 calculations.
        """
        mlwf = MLWFReaderQeW90(mlwf_setting, if_copy = False)
        mlwf.default()
        self.scf_writers: Dict[str, QeParamsConfs] = {}
        if mlwf.run_nscf:
            self.nscf_writers: Dict[str, QeParamsConfs] = {}
        self.pw2wan_writers: Dict[str, Dict[str, QeParams]] = {}
        self.wannier90_writers: Dict[str, Dict[str, Wannier90Inputs]] = {}
        self.get_writers(mlwf, confs, wc_python)
        self.mlwf = mlwf
        return mlwf.mlwf_setting

    def prep_one_frame(self, frame: int):
        seed_name = self.mlwf.seed_name
        for qe_key in self.scf_writers:
            with set_directory(qe_key, mkdir = True):
                Path(f"scf_{qe_key}.in").write_text(self.scf_writers[qe_key].write(frame))
                if self.mlwf.run_nscf:
                    Path(f"nscf_{qe_key}.in").write_text(self.nscf_writers[qe_key].write(frame))
                for w90_key in self.pw2wan_writers[qe_key]:
                    Path(f"{seed_name(qe_key, w90_key)}.pw2wan").write_text(self.pw2wan_writers[qe_key][w90_key].write(frame))
                    Path(f"{seed_name(qe_key, w90_key)}.win").write_text(self.wannier90_writers[qe_key][w90_key].write(frame))

class RunQeWann(RunMLWF):
    DEFAULT_BACK = ["*.xyz"]
    def __init__(self) -> None:
        super().__init__()

    def init_cmd(self, mlwf_setting: Dict[str, Union[str, dict]], commands: Dict[str, str]):
        self.mlwf = MLWFReaderQeW90(mlwf_setting)
        self.pw_cmd = commands.get("pw", "pw.x")
        self.pw2wan_cmd = commands.get("pw2wannier", "pw2wannier90.x")
        self.wannier_cmd = commands.get("wannier90", "wannier90.x")
    
    def run_one_subtask(
            self, qe_key, back: Path, out_dir: Path, 
            copy_out: bool = False, tar_dir: Path = None
        ):
        mlwf = self.mlwf
        scf_name = mlwf.scf_name(qe_key)
        nscf_name = mlwf.nscf_name(qe_key)
        self.run(" ".join([self.pw_cmd, "-input", scf_name]))
        if copy_out:
            shutil.copytree(out_dir, tar_dir)
        if Path(nscf_name).exists():
            self.run(" ".join([self.pw_cmd, "-input", nscf_name]))
        for w90_key in mlwf.get_w90_params_dict():
            seed_name = mlwf.seed_name(qe_key, w90_key)
            self.run(" ".join([self.wannier_cmd, "-pp", seed_name]))
            self.run(" ".join([self.pw2wan_cmd]), input=Path(f"{seed_name}.pw2wan").read_text())
            self.run(" ".join([self.wannier_cmd, seed_name]))
            shutil.copy(f"{seed_name}_centres.xyz", back)
        shutil.rmtree(out_dir)

    def run_one_frame(self, backward_dir_name: str) -> Path:
        mlwf = self.mlwf
        out_dir = Path(mlwf.scf_params["control"]["outdir"])
        ori_out_dir = Path("ori_out_temp").absolute()
        ori_p = Path("ori")
        backward_dir = Path(backward_dir_name)
        backward_dir.mkdir()
        back_abs = backward_dir.absolute()

        with_ef = mlwf.with_efield
        if with_ef:
            ef_type = mlwf.ef_type

        with set_directory(ori_p):
            self.run_one_subtask(
                "ori", back_abs, out_dir, with_ef and ef_type == "enthalpy", ori_out_dir
            )
        if with_ef:
            for ef_name in mlwf.efields:
                qe_key = f"ef_{ef_name}"
                with set_directory(qe_key):
                    if ef_type == "enthalpy":
                        shutil.copytree(ori_out_dir, out_dir)
                    self.run_one_subtask(qe_key, back_abs, out_dir)
        
        return backward_dir

class CollectWann(CollectWFC):
    def init_params(self, mlwf_setting: dict, conf_sys: dpdata.System, example_file: Path):
        self.mlwf = MLWFReaderQeW90(mlwf_setting)
        self.nat = len(conf_sys["atom_types"])

    def get_one_frame(self) -> Dict[str, np.ndarray]:
        mlwf = self.mlwf
        wfc_dict = {}
        for qe_key in mlwf.qe_iter:
            wfc_l = []
            for w90_key in mlwf.get_w90_params_dict():
                wfc_l.append(self.read_wfc(mlwf.seed_name(qe_key, w90_key)))
            wfc_dict[qe_key] = np.concatenate(wfc_l, axis = 0)
        return wfc_dict
    
    def read_wfc(self, seed_name: str) -> np.ndarray:
        with open(f'{seed_name}_centres.xyz', "r") as f:
            num_wann = int(np.loadtxt(f, dtype = int, max_rows = 1)) - self.nat
            wfc = np.loadtxt(f, dtype = float, skiprows = 1, usecols = [1, 2, 3], max_rows = num_wann, ndmin = 2)
        return wfc

