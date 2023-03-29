from typing import Dict, Union
from pathlib import Path
import dpdata, time
from dflow.plugins.dispatcher import DispatcherExecutor
from dflow import (
    Step, 
    Workflow,
    upload_artifact,
    download_artifact
)
from dflow.python import (
    PythonOPTemplate,
)
from mlwf_op.prepare_input_op import Prepare
from mlwf_op.inputs import QeParamsConfs, QeParams, Wannier90Inputs, complete_qe, complete_wannier90, complete_pw2wan
from mlwf_op.utils import complete_by_default

class PrepareQe(Prepare):
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

    def init_inputs(self, input_setting: Dict[str, Union[str, dict]], confs: dpdata.System):
        self.name = input_setting["name"]
        self.run_nscf = input_setting["dft_params"]["cal_type"] == "scf+nscf"

        k_grid = input_setting["dft_params"]["k_grid"]
        qe_params = complete_by_default(input_setting["dft_params"]["qe_params"], params_default = self.DEFAULT_PARAMS, if_copy = True)

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
        wan_params, proj, kpoints = complete_wannier90(
            input_setting["wannier90_params"]["wan_params"], 
            input_setting["wannier90_params"]["projections"],
            input_setting["dft_params"]["k_grid"]
        )
        self.wannier90_writer = Wannier90Inputs(wan_params, proj, kpoints, confs)
        return super().init_inputs(input_setting, confs)

    def write_one_frame(self, frame: int):
        Path("scf.in").write_text(self.scf_writer.write(frame))
        if self.run_nscf:
            Path("nscf.in").write_text(self.nscf_writer.write(frame))
        Path(f"{self.name}.pw2wan").write_text(self.pw2wan_writer.write(frame))
        Path(f"{self.name}.win").write_text(self.wannier90_writer.write(frame))
        return super().write_one_frame(frame)