from typing import Dict, List, Tuple, Union
from pathlib import Path
import abc
import shutil
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    PythonOPTemplate,
    Slices, 
    BigParameter,
    upload_packages
)
from dflow.utils import (
    set_directory,
    run_command
)

class RunMLWF(OP, abc.ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "task_path": Artifact(Path),
            "input_setting": BigParameter(dict),
            "task_setting": BigParameter(dict),
            "frames": BigParameter(Tuple[int]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "backward": Artifact(List[Path])
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        task_path: Path = op_in["task_path"]
        self.name: str = op_in["input_setting"]["name"]
        task_setting: dict = op_in["task_setting"]
        backward_list: List[str] = task_setting["backward_list"]
        backward_dir_name: str = task_setting["backward_dir_name"]
        commands: Dict[str, str] = task_setting["commands"]
        start_f, end_f = op_in["frames"]

        confs_path = [Path(f"conf.{f:06d}") for f in range(start_f, end_f)]
        self.log_path = Path("run.log")

        self.init_cmd(commands)
        backward = self._exec_all(task_path, confs_path, backward_dir_name, backward_list)
        return OPIO({
            "backward": backward
        })
    
    def _exec_all(self, task_path: Path, confs_path: List[Path], backward_dir_name: str, backward_list: List[str]):
        backward: List[Path] = []
        with set_directory(task_path):
            for p in confs_path:
                with set_directory(p):
                    backward_dir = self.run_one_frame(backward_dir_name, backward_list)
                    backward.append(task_path / p / backward_dir)
        return backward

    def run(self, *args, **kwargs):
        if_print = True
        if "if_print" in kwargs:
            if_print = kwargs["if_print"]
            del kwargs["if_print"]
        ret, out, err = run_command(*args, **kwargs)
        if if_print:
            print(out)
        # Save log here.
        with self.log_path.open(mode = "a") as fp:
            fp.write(out)

    @abc.abstractmethod
    def init_cmd(self, commands: Dict[str, str]):
        pass

    @abc.abstractmethod
    def run_one_frame(self, backward_dir_name: str, backward_list: List[str]) -> Path:
        pass