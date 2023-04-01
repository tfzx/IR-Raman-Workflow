from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from mlwf_op.run_mlwf_op import RunMLWF
import dpdata
from mlwf_op.prepare_input_op import Prepare
from mlwf_op.inputs import QeParamsConfs, QeParams, Wannier90Inputs, complete_qe, complete_wannier90, complete_pw2wan
from mlwf_op.utils import complete_by_default


class PrepareCP(Prepare):
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
        qe_params = complete_by_default(input_setting["dft_params"]["qe_params"], params_default = self.DEFAULT_PARAMS)
        input_cp, _ = complete_qe(qe_params, "cp-wf", None, confs)
        self.cp_writer = QeParamsConfs(input_cp, None, input_setting["dft_params"]["atomic_species"], confs)
        return input_setting

    def write_one_frame(self, frame: int):
        Path(f"cp_{self.name}.in").write_text(self.cp_writer.write(frame))
        return super().write_one_frame(frame)

class RunCPWF(RunMLWF):
    def __init__(self) -> None:
        super().__init__()
    
    def init_cmd(self, commands: Dict[str, str]):
        self.cp_cmd = commands.get("cp", "cp.x")
        return super().init_cmd(commands)

    def run_one_frame(self) -> Path:
        self.run(" ".join([self.cp_cmd, "-input", f"cp_{self.name}.in"]))
        backward_dir = Path(self.backward_dir_name)
        backward_dir.mkdir()
        return backward_dir