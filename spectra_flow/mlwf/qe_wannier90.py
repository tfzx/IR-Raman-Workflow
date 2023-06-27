import shutil
from types import ModuleType
from typing import Dict, Optional, Union
from pathlib import Path
import dpdata, numpy as np
from dflow.utils import set_directory
from spectra_flow.mlwf.mlwf_ops import Prepare, RunMLWF, CollectWFC
from spectra_flow.mlwf.inputs import (
    QePwInputs, 
    QePw2wanInputs, 
    Wannier90Inputs, 
    get_qe_writers,
    get_pw_w90_writers
)
from spectra_flow.SuperOP.mlwf_reader import MLWFReaderQeW90

class PrepareQeWann(Prepare):
    """
    Description
    -----
    Generate input files for Qe-Pwscf and Wannier90. 
    Support calculation with electric fields.

    Generate
    -----
    Prepare the inputs file for each frame.
    There might be multiple qe runs (for example, add different electric fields),
    and each qe run will generate a directory to place the corresponding input files.
    -----
    The input files for each qe run:
        - `scf_{qe_key}.in`
        - `nscf_{qe_key}.in` (optional. If `cal_type == "scf+nscf"`, this file will be generated.)
        - `{seed_name}.pw2wan` 
        - `{seed_name}.win`

    Notice that if there are multiple wannier90 runs (defined by `multi_w90_params`), 
    each dir will generate a `seed_name` and the corresponding `.pw2wan` and `.win` files.
    -----
    An example of the file structure:
    ```txt
    task.000000
    `-- conf.000000
        |-- ori
        |   |-- scf_ori.in
        |   |-- nscf_ori.in
        |   |-- graphene_ori_pz.pw2wan
        |   |-- graphene_ori_pz.win
        |   |-- graphene_ori_sp2.pw2wan
        |   `-- graphene_ori_sp2.win
        `-- ef_x
    ```
    """
    @classmethod
    def get_w90_rewriter(cls, wc_python: Optional[ModuleType] = None):
        """Try to import functions `rewrite_atoms` and `rewrite_proj` from `wc_python` and return them."""
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

    def get_writers(self, mlwf: MLWFReaderQeW90, confs: dpdata.System, wc_python: Optional[ModuleType]):
        """
        Description
        -----
        Get `scf_writers`, `nscf_writers`, `pw2wan_writers` and `wannier90_writers` from `mlwf`,
        and save them as the attributes.

        Parameters:
        -----
        mlwf: `MLWFReaderQeW90`. The `mlwf_setting` wrapper.

        confs: `dpdata.System`. The system.

        wc_python: `ModuleType`, optional.
            If provided, this will try to import functions `rewrite_atoms` and `rewrite_proj` from it.
            See `spectra_flow.mlwf.inputs.Wannier90Inputs` for details.

        Attributes:
        -----
        All the attributes below was defined in method `init_inputs`.
        - scf_writers: `Dict[str, QeParamsConfs]`
        - nscf_writers: `Dict[str, QeParamsConfs]`
        - pw2wan_writers: `Dict[str, Dict[str, QeParams]]`
        - wannier90_writers: `Dict[str, Dict[str, Wannier90Inputs]]`
        """
        w90_params_dict = mlwf.get_w90_params_dict()
        scf_grid, nscf_grid = mlwf.get_kgrid()
        w90_kgrid = nscf_grid if mlwf.run_nscf else scf_grid
        nscf_params = mlwf.nscf_params
        pw2wan_params = mlwf.pw2wan_params
        atomic_species = mlwf.atomic_species
        assert atomic_species is not None, "atomic_species is None!"
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
                assert nscf_writer is not None
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
                    rewrite_atoms, rewrite_proj # type: ignore
                )

    def init_inputs(self, mlwf_setting: Dict[str, Union[str, dict]], confs: dpdata.System, wc_python: Optional[ModuleType]):
        mlwf = MLWFReaderQeW90(mlwf_setting, if_copy = False, if_print = True)
        self.scf_writers: Dict[str, QePwInputs] = {}
        if mlwf.run_nscf:
            self.nscf_writers: Dict[str, QePwInputs] = {}
        self.pw2wan_writers: Dict[str, Dict[str, QePw2wanInputs]] = {}
        self.wannier90_writers: Dict[str, Dict[str, Wannier90Inputs]] = {}
        self.get_writers(mlwf, confs, wc_python)
        self.mlwf = mlwf
        return mlwf.mlwf_setting

    def prep_one_frame(self, frame: int):
        """
            Each dir runs one qe calculations (scf+[nscf]) and some wannier90 calculations.
        """
        mlwf = self.mlwf
        for qe_key in self.scf_writers:
            with set_directory(qe_key, mkdir = True): # type: ignore
                Path(mlwf.scf_name(qe_key)).write_text(self.scf_writers[qe_key].write(frame))
                if mlwf.run_nscf:
                    Path(mlwf.nscf_name(qe_key)).write_text(self.nscf_writers[qe_key].write(frame))
                for w90_key in self.pw2wan_writers[qe_key]:
                    seed_name = mlwf.seed_name(qe_key, w90_key)
                    Path(f"{seed_name}.pw2wan").write_text(self.pw2wan_writers[qe_key][w90_key].write(frame))
                    Path(f"{seed_name}.win").write_text(self.wannier90_writers[qe_key][w90_key].write(frame))

class RunQeWann(RunMLWF):
    """
    Description
    -----
    Run Qe-Pwscf and Wannier90. 
    Support calculation with electric fields.

    See Also
    -----
    `spectra_flow.mlwf.qe_wannier90.PrepareQeWann`
    """
    DEFAULT_BACK = []
    def init_cmd(self, mlwf_setting: Dict[str, Union[str, dict]], commands: Dict[str, str]):
        self.mlwf = MLWFReaderQeW90(mlwf_setting)
        self.pw_cmd = commands.get("pw", "pw.x")
        self.pw2wan_cmd = commands.get("pw2wannier", "pw2wannier90.x")
        self.wannier_cmd = commands.get("wannier90", "wannier90.x")
    
    def run_one_subtask(
            self, qe_key, back: Path, out_dir: Path, 
            copy_out: bool = False, tar_dir: Optional[Path] = None
        ):
        """
            Each dir runs one qe calculations (scf+[nscf]) and some wannier90 calculations.
        """
        mlwf = self.mlwf
        scf_name = mlwf.scf_name(qe_key)
        nscf_name = mlwf.nscf_name(qe_key)
        assert Path(scf_name).exists(), f"Qe scf input file '{scf_name}' doesn't exist!"
        self.run(" ".join([self.pw_cmd, "-input", scf_name]))
        if copy_out and tar_dir:
            shutil.copytree(out_dir, tar_dir)
        if Path(nscf_name).exists():
            self.run(" ".join([self.pw_cmd, "-input", nscf_name]))
        for w90_key in mlwf.get_w90_params_dict():
            seed_name = mlwf.seed_name(qe_key, w90_key)
            assert Path(f"{seed_name}.win").exists(), f"Wannier90 input file '{seed_name}.win' doesn't exist!"
            assert Path(f"{seed_name}.pw2wan").exists(), f"Qe pw2wannier90 input file '{seed_name}.pw2wan' doesn't exist!"
            self.run(" ".join([self.wannier_cmd, "-pp", seed_name]))
            self.run(" ".join([self.pw2wan_cmd]), input=Path(f"{seed_name}.pw2wan").read_text())
            self.run(" ".join([self.wannier_cmd, seed_name]), raise_error = False)
            shutil.copy(f"{seed_name}_centres.xyz", back)
        shutil.rmtree(out_dir)

    def run_one_frame(self, backward_dir: Path):
        mlwf = self.mlwf
        out_dir = Path(mlwf.scf_params["control"]["outdir"]) # type: ignore
        ori_out_dir = Path("ori_out_temp").absolute()
        ori_p = Path("ori")
        back_abs = backward_dir.absolute()

        with_ef = mlwf.with_efield
        ef_type = mlwf.ef_type

        with set_directory(ori_p):
            self.run_one_subtask(
                "ori", back_abs, out_dir, with_ef and ef_type == "enthalpy", ori_out_dir
            )
        if with_ef:
            for ef_name in mlwf.efields: # type: ignore
                qe_key = f"ef_{ef_name}"
                with set_directory(qe_key): # type: ignore
                    if ef_type == "enthalpy":
                        try:
                            shutil.copytree(ori_out_dir, out_dir)
                        except Exception as e:
                            print(f"[WARNING] Cannot copy the outdir to restart: {e}")
                            if self.debug:
                                raise e
                    self.run_one_subtask(qe_key, back_abs, out_dir)

class CollectWann(CollectWFC):
    """
    Description
    -----
    Collect wfc from `*_centres.xyz` generated by Wannier90. 
    Support calculation with electric fields.

    See Also
    -----
    `spectra_flow.mlwf.qe_wannier90.PrepareQeWann`
    """
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

