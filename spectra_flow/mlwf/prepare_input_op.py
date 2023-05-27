from math import ceil
from types import ModuleType
from typing import Dict, List, Tuple, Union
from pathlib import Path
import abc
import shutil
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter
)
from dflow.utils import (
    set_directory
)
import dpdata
from spectra_flow.utils import read_conf


class Prepare(OP, abc.ABC):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "input_setting": BigParameter(dict),
            "task_setting": BigParameter(dict),
            "confs": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "cal_dipole_python": Artifact(Path, optional = True),
            "pseudo": Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "input_setting": BigParameter(dict),
            "task_path": Artifact(List[Path], archive=None),
            "frames_list": BigParameter(List[Tuple[int]])
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        input_setting: Dict[str, Union[str, dict]] = op_in["input_setting"]
        group_size: int = op_in["task_setting"]["group_size"]
        confs = read_conf(op_in["confs"], op_in["conf_fmt"])
        pseudo: Path = op_in["pseudo"]
        if op_in["cal_dipole_python"]:
            import imp
            wc_python = imp.load_source("dipole_module", str(op_in["cal_dipole_python"]))
        else:
            wc_python = None

        input_setting = self.init_inputs(input_setting, confs, wc_python)
        task_path, frames_list = self._exec_all(confs, pseudo, group_size)
        return OPIO({
            "input_setting": input_setting,
            "task_path": task_path,
            "frames_list": frames_list
        })

    def _exec_all(self, confs: dpdata.System, pseudo: Path, group_size: int):
        nframes = confs.get_nframes()
        task_path = []
        frames_list = []
        for i in range(ceil(nframes / group_size)):
            task_name = f"task.{i:06d}"
            print(task_name)
            task = Path(task_name)
            task_path.append(task)
            start_f = i * group_size
            end_f = min(start_f + group_size, nframes)
            frames_list.append((start_f, end_f))
            with set_directory(task_name, mkdir = True):
                shutil.copytree(pseudo, "./pseudo")
                for frame in range(start_f, end_f):
                    subtask_name = f"conf.{frame:06d}"
                    print(subtask_name)
                    with set_directory(subtask_name, mkdir = True):
                        self.write_one_frame(frame)
        return task_path, frames_list

    @abc.abstractmethod
    def init_inputs(self, 
                    input_setting: Dict[str, Union[str, dict]], 
                    confs: dpdata.System,
                    wc_python: ModuleType = None) -> Dict[str, Union[str, dict]]:
        pass

    @abc.abstractmethod
    def write_one_frame(self, frame: int):
        pass
