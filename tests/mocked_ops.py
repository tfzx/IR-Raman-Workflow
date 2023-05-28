from typing import Dict, List, Tuple, Union
from spectra_flow.mlwf.mlwf_ops import Prepare, RunMLWF
from pathlib import Path
import dpdata, os

class MockedPrepare(Prepare):
    def init_inputs(self, mlwf_setting: Dict[str, Union[str, dict]], confs: dpdata.System):
        self.task_path = []
        self.conf_path = []
        return super().init_inputs(mlwf_setting, confs)
    
    def write_one_frame(self, frame: int):
        cwd = Path(os.getcwd())
        self.conf_path.append(cwd.name)
        self.task_path.append(cwd.parent.name)
        return super().write_one_frame(frame)

class MockedRunMLWF(RunMLWF):
    def init_cmd(self, commands: Dict[str, str]):
        return super().init_cmd(commands)

    def run_one_frame(self) -> Path:
        return super().run_one_frame()