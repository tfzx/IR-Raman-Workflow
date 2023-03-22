from typing import Dict, List, Tuple, Union
from mlwf_op.prepare_input_op import Prepare
from pathlib import Path
import dpdata, os

class MockedPrepare(Prepare):
    def init_inputs(self, input_setting: Dict[str, Union[str, dict]], confs: dpdata.System):
        self.task_path = []
        self.conf_path = []
        return super().init_inputs(input_setting, confs)
    
    def write_one_frame(self, frame: int):
        cwd = Path(os.getcwd())
        self.conf_path.append(cwd.name)
        self.task_path.append(cwd.parent.name)
        return super().write_one_frame(frame)