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
    set_directory
)

class RunMLWF(OP, abc.ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "name": str,
            "backward_dir_name": str,
            "backward_list": BigParameter(List[str]),
            "commands": BigParameter(Dict[str, str]),
            "task_path": Artifact(Path),
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
        self.name: str = op_in["name"]
        backward_list: List[str] = op_in["backward_list"]
        backward_dir_name: str = op_in["backward_dir_name"]
        commands: Dict[str, str] = op_in["commands"]
        task_path: Path = op_in["task_path"]
        start_f, end_f = op_in["frames"]
        confs_path = [Path(f"conf.{f:06d}") for f in range(start_f, end_f)]

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

    @abc.abstractmethod
    def init_cmd(self, commands: Dict[str, str]):
        pass

    @abc.abstractmethod
    def run_one_frame(self, backward_dir_name: str, backward_list: List[str]) -> Path:
        pass