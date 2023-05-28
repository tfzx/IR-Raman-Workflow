from math import ceil
from types import ModuleType
from typing import Dict, List, Tuple, Union
from pathlib import Path
import abc, shutil, numpy as np
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter
)
from dflow.utils import (
    set_directory,
    run_command
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
                        self.prep_one_frame(frame)
        return task_path, frames_list

    @abc.abstractmethod
    def init_inputs(self, 
                    input_setting: Dict[str, Union[str, dict]], 
                    confs: dpdata.System,
                    wc_python: ModuleType = None) -> Dict[str, Union[str, dict]]:
        pass

    @abc.abstractmethod
    def prep_one_frame(self, frame: int):
        pass

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
        self.input_setting = op_in["input_setting"]
        task_setting: dict = op_in["task_setting"]
        self.backward_list: List[str] = task_setting["backward_list"]
        self.backward_dir_name: str = task_setting["backward_dir_name"]
        commands: Dict[str, str] = task_setting["commands"]
        start_f, end_f = op_in["frames"]

        confs_path = [Path(f"conf.{f:06d}") for f in range(start_f, end_f)]
        self.log_path = Path("run.log").absolute()

        self.init_cmd(commands)
        backward = self._exec_all(task_path, confs_path)
        return OPIO({
            "backward": backward
        })
    
    def _exec_all(self, task_path: Path, confs_path: List[Path]):
        backward: List[Path] = []
        with set_directory(task_path):
            for p in confs_path:
                with set_directory(p):
                    backward_dir = self.run_one_frame()
                    self._collect(backward_dir)
                    backward.append(task_path / p / backward_dir)
        return backward

    def _collect(self, backward_dir: Path):
        for f in self.backward_list:
            for p in Path(".").glob(f):
                print(p.name)
                if p.is_file():
                    shutil.copy(p, backward_dir)
                else:
                    shutil.copytree(p, backward_dir / p.name)
        shutil.copy(self.log_path, backward_dir)

    def run(self, *args, **kwargs):
        kwargs["print_oe"] = True
        kwargs["raise_error"] = True
        ret, out, err = run_command(*args, **kwargs)
        # if ret != 0:
        #     print(out)
        #     print(err)
        #     assert ret == 0
        # Save log here.
        with self.log_path.open(mode = "a") as fp:
            fp.write(out)

    @abc.abstractmethod
    def init_cmd(self, commands: Dict[str, str]):
        pass

    @abc.abstractmethod
    def run_one_frame(self) -> Path:
        pass

class CollectWFC(OP, abc.ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "input_setting": BigParameter(dict),
            "confs": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "backward": Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "wannier_function_centers": Artifact(Dict[str, Path])
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        input_setting: dict = op_in["input_setting"]
        conf_sys = read_conf(op_in["confs"], op_in["conf_fmt"])
        backward: List[Path] = op_in["backward"]

        wfc_path = self.collect_wfc(input_setting, conf_sys, backward)
        return OPIO({
            "wannier_function_centers": wfc_path
        })
    
    def collect_wfc(self, input_setting: dict, conf_sys: dpdata.System, backward: List[Path]):
        assert len(backward) > 0
        self.init_params(input_setting, conf_sys, backward)
        total_wfc: Dict[str, np.ndarray] = {}
        def update_wfc(wfc_frame: Dict[str, np.ndarray], frame: int):
            for key, wfc_arr in wfc_frame.items():
                if key not in total_wfc:
                    total_wfc[key] = np.zeros((conf_sys.get_nframes(), wfc_arr.size), dtype = float)
                total_wfc[key][frame] = wfc_arr.flatten()
        for frame, p in enumerate(backward):
            with set_directory(p):
                update_wfc(self.get_one_frame(frame), frame)
        wfc_path: Dict[str, Path] = {}
        for key, wfc in total_wfc.items():
            wfc_path[key] = Path(f"wfc_{key}.raw")
            np.savetxt(wfc_path[key], wfc)
        return wfc_path


    @abc.abstractmethod
    def get_one_frame(self, frame: int) -> Dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def init_params(self, input_setting: dict, conf_sys: dpdata.System, example_file: Path):
        try:
            self.num_wann = input_setting["num_wann"]
        except KeyError:
            self.num_wann = self.get_num_wann(conf_sys, example_file)

    @abc.abstractmethod
    def get_num_wann(self, conf_sys: dpdata.System, example_file: Path) -> int:
        pass