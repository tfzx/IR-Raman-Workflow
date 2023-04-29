from typing import Dict, List, Union
from pathlib import Path
import dpdata, numpy as np
from spectra_flow.mlwf.prepare_input_op import Prepare
from spectra_flow.mlwf.run_mlwf_op import RunMLWF
from spectra_flow.mlwf.collect_wfc_op import CollectWFC
from spectra_flow.mlwf.inputs import QeParamsConfs, complete_qe
from spectra_flow.mlwf.utils import complete_by_default


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

class CollectCPWF(CollectWFC):
    a0 = 0.5291772083
    def init_params(self, input_setting: dict, backward: List[Path]):
        self.prefix = input_setting["dft_params"]["qe_params"]["control"]["prefix"]
        return super().init_params(input_setting, backward)

    def get_num_wann(self, file_path: Path) -> int:
        start = False
        num_wann = 0
        with open(file_path / f"{self.prefix}.wfc", "r") as fp:
            for line in fp.readlines():
                line = line.strip()
                if not line:
                    break
                if len(line.split()) > 3:
                    if start:
                        break
                    else:
                        start = True
                elif start:
                    num_wann += 1
        return num_wann

    def get_one_frame(self, frame: int) -> np.ndarray:
        with open(f"{self.prefix}.wfc", "r") as fp:
            while fp.readline().strip():
                wann = np.loadtxt(fp, dtype = float, max_rows = self.num_wann)
        return wann * self.a0