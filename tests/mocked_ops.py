from types import ModuleType
from typing import Dict, List, Optional, Tuple, Union
import shutil
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
    DEFAULT_BACK = ["*.out", "*.xyz"]
    def __init__(
            self, 
            success_list_test1: List[bool],
            success_list_test2: List[bool],
            success_list_test3: List[bool],
        ) -> None:
        self.success_list_test1 = success_list_test1
        self.success_list_test2 = success_list_test2
        self.success_list_test3 = success_list_test3
        super().__init__()


    def init_cmd(self, mlwf_setting: Dict[str, Union[str, dict]], commands: Dict[str, str]):
        self.test_flag = True
        return super().init_cmd(mlwf_setting, commands)

    def run_one_frame(self, backward_dir: Path):
        assert self.test_flag
        p1 = Path("test1")
        p1.write_text("test1")
        flag1 = self.success_list_test1.pop(0)
        flag2 = self.success_list_test2.pop(0)
        flag3 = self.success_list_test3.pop(0)
        assert flag1, "Test: fail after test1"
        p2 = Path("test2.out")
        p2.write_text("test2\ntest2")
        assert flag2, "Test: fail after test2.out"
        p3 = Path("test3.xyz")
        p3.write_text("test3\ntest3\ntest3")
        assert flag3, "Test: fail after test3.xyz"
        # Explicitly collect test1 and test2.out
        # test3.xyz should be collected by DEFAULT_BACK
        # test2.out will be collected again by DEFAULT_BACK
        shutil.copy(p1, backward_dir / p1)
        shutil.copy(p2, backward_dir / p2)
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