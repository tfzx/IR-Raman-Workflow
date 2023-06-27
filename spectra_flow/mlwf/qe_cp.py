from typing import Dict, List, Optional, Union
from types import ModuleType
from pathlib import Path
import dpdata, numpy as np
from spectra_flow.mlwf.mlwf_ops import Prepare, RunMLWF, CollectWFC
from spectra_flow.mlwf.inputs import QeCPInputs, complete_qe
from spectra_flow.utils import complete_by_default


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
    
    def init_inputs(self, 
                    mlwf_setting: Dict[str, Union[str, dict]], 
                    confs: dpdata.System,
                    wc_python: Optional[ModuleType] = None) -> Dict[str, Union[str, dict]]:
        self.name = mlwf_setting["name"]
        qe_params = complete_by_default(mlwf_setting["dft_params"]["qe_params"], params_default = self.DEFAULT_PARAMS) # type: ignore
        input_cp, _ = complete_qe(qe_params, "cp-wf", None, confs)
        self.cp_writer = QeCPInputs(input_cp, None, mlwf_setting["dft_params"]["atomic_species"], confs) # type: ignore
        return mlwf_setting

    def prep_one_frame(self, frame: int):
        Path(f"cp_{self.name}.in").write_text(self.cp_writer.write(frame))

class RunCPWF(RunMLWF):
    DEFAULT_BACK = ["*/*.wfc"]
    def __init__(self) -> None:
        super().__init__()
    
    def init_cmd(self, mlwf_setting: Dict[str, Union[str, dict]], commands: Dict[str, str]):
        self.cp_cmd = commands.get("cp", "cp.x")
        self.name = mlwf_setting["name"]
        return super().init_cmd(mlwf_setting, commands)

    def run_one_frame(self, backward_dir: Path):
        self.run(" ".join([self.cp_cmd, "-input", f"cp_{self.name}.in"]))

class CollectCPWF(CollectWFC):
    a0 = 0.5291772083
    def init_params(self, mlwf_setting: dict, conf_sys: dpdata.System, example_file: Path):
        self.prefix = mlwf_setting["dft_params"]["qe_params"]["control"]["prefix"]
        try:
            self.num_wann = mlwf_setting["num_wann"]
        except KeyError:
            self.num_wann = self.get_num_wann(example_file)

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

    def get_one_frame(self) -> Dict[str, np.ndarray]:
        wann: Optional[np.ndarray] = None
        with open(f"{self.prefix}.wfc", "r") as fp:
            while fp.readline().strip():
                wann = np.loadtxt(fp, dtype = float, max_rows = self.num_wann)
        assert wann is not None
        return {"ori": wann * self.a0} # type: ignore