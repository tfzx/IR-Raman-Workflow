from types import ModuleType
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from spectra_flow.mlwf.mlwf_ops import CollectWFC, Prepare, RunMLWF
from pathlib import Path
import dpdata, os

class MockedPrepare(Prepare):
    def init_inputs(self, 
                    mlwf_setting: Dict[str, Union[str, dict]], 
                    confs: dpdata.System,
                    wc_python: Optional[ModuleType] = None) -> Dict[str, Union[str, dict]]:
        self.task_path = []
        self.conf_path = []
        return super().init_inputs(mlwf_setting, confs)
    
    def prep_one_frame(self, frame: int):
        cwd = Path(os.getcwd())
        self.conf_path.append(cwd.name)
        self.task_path.append(cwd.parent.name)
        return super().prep_one_frame(frame)

class MockedRunMLWF(RunMLWF):
    def init_cmd(self, mlwf_setting: Dict[str, Union[str, dict]], commands: Dict[str, str]):
        return super().init_cmd(mlwf_setting, commands)

    def run_one_frame(self, backward_dir: Path):
        return super().run_one_frame(backward_dir)
    
class MockedCollectWFC(CollectWFC):
    def __init__(self, success_list: Optional[List[bool]] = None) -> None:
        self.success_list = success_list
        super().__init__()

    def init_params(self, mlwf_setting: dict, conf_sys: dpdata.System, example_file: Path):
        self.confs = conf_sys

    def get_one_frame(self) -> Dict[str, np.ndarray]:
        if self.success_list is not None:
            if not self.success_list.pop(0):
                raise RuntimeError("Test: raise error.")
        return {
            "test1": np.zeros((10, 3)),
            "test2": np.ones((9, 3))
        }