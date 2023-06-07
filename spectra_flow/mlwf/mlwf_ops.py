from math import ceil
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Union
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
from spectra_flow.utils import read_conf, dump_to_fmt


class Prepare(OP, abc.ABC):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "mlwf_setting": BigParameter(dict),
            "task_setting": BigParameter(dict),
            "confs": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "cal_dipole_python": Artifact(Path, optional = True),
            "pseudo": Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "mlwf_setting": BigParameter(dict),
            "task_path": Artifact(List[Path], archive=None), # type: ignore
            "frames_list": BigParameter(List[Tuple[int]])
        })

    @OP.exec_sign_check # type: ignore
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        mlwf_setting: Dict[str, Union[str, dict]] = op_in["mlwf_setting"]
        group_size: int = op_in["task_setting"]["group_size"]
        confs = read_conf(op_in["confs"], op_in["conf_fmt"])
        pseudo: Path = op_in["pseudo"]
        if op_in["cal_dipole_python"]:
            import imp
            wc_python = imp.load_source("dipole_module", str(op_in["cal_dipole_python"]))
        else:
            wc_python = None

        mlwf_setting = self.init_inputs(mlwf_setting, confs, wc_python)
        task_path, frames_list = self._exec_all(confs, pseudo, group_size)
        return OPIO({
            "mlwf_setting": mlwf_setting,
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
            with set_directory(task_name, mkdir = True): # type: ignore
                shutil.copytree(pseudo, "./pseudo")
                for frame in range(start_f, end_f):
                    subtask_name = f"conf.{frame:06d}"
                    print(subtask_name)
                    with set_directory(subtask_name, mkdir = True): # type: ignore
                        self.prep_one_frame(frame)
        return task_path, frames_list

    @abc.abstractmethod
    def init_inputs(self, 
                    mlwf_setting: Dict[str, Union[str, dict]], 
                    confs: dpdata.System,
                    wc_python: Optional[ModuleType] = None) -> Dict[str, Union[str, dict]]:
        pass

    @abc.abstractmethod
    def prep_one_frame(self, frame: int):
        pass

class RunMLWF(OP, abc.ABC):
    DEFAULT_BACK = []
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "task_path": Artifact(Path),
            "mlwf_setting": BigParameter(dict),
            "task_setting": BigParameter(dict),
            "frames": BigParameter(Tuple[int]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "backward": Artifact(List[Path])
        })

    @OP.exec_sign_check # type: ignore
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        task_path: Path = op_in["task_path"]
        mlwf_setting: dict = op_in["mlwf_setting"]
        task_setting: dict = op_in["task_setting"]
        _backward_list = task_setting.get("backward_list", [])
        backward_list: List[str] = list(set(_backward_list + self.DEFAULT_BACK))
        backward_dir_name: str = task_setting.get("backward_dir_name", "back")
        commands: Dict[str, str] = task_setting.get("commands", {})
        start_f, end_f = op_in["frames"]

        self.init_cmd(mlwf_setting, commands)
        backward = self._exec_all(task_path, start_f, end_f, backward_list, backward_dir_name)
        return OPIO({
            "backward": backward
        })
    
    def _exec_all(self, task_path: Path, start_f: int, end_f: int, backward_list: List[str], backward_dir_name: str):
        backward: List[Path] = []
        with set_directory(task_path):
            for frame in range(start_f, end_f):
                conf_path = Path(f"conf.{frame:06d}")
                with set_directory(conf_path):
                    self.log_path = Path("run.log").absolute()
                    self.log_path.touch()

                    backward_dir = Path(backward_dir_name)
                    backward_dir.mkdir()
                    try:
                        self.run_one_frame(backward_dir)
                    except Exception as e:
                        print(f"[ERROR] Frame {frame:06d} failed: {e}")
                    self._collect(backward_dir, backward_list)
                    backward.append(task_path / conf_path / backward_dir)
        return backward

    def _collect(self, backward_dir: Path, backward_list: List[str]):
        for f in backward_list:
            for p in Path(".").glob(f):
                print(p.name)
                if p.is_file():
                    shutil.copy(p, backward_dir)
                else:
                    shutil.copytree(p, backward_dir / p.name)
        shutil.copy(self.log_path, backward_dir)

    def run(self, *args, **kwargs):
        kwargs["print_oe"] = True
        kwargs["raise_error"] = False
        ret, out, err = run_command(*args, **kwargs)
        if ret != 0:
            print(f"[WARNING] Exception return of command {kwargs['cmd']}: \n{err}")
        #     assert ret == 0
        # Save log here.
        with self.log_path.open(mode = "a") as fp:
            fp.write(out)

    @abc.abstractmethod
    def init_cmd(self, mlwf_setting: Dict[str, Union[str, dict]], commands: Dict[str, str]):
        pass

    @abc.abstractmethod
    def run_one_frame(self, backward_dir: Path):
        pass

class CollectWFC(OP, abc.ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "mlwf_setting": BigParameter(dict),
            "confs": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "backward": Artifact(List[Path])
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "final_confs": Artifact(Path),
            "final_conf_fmt": BigParameter(dict),
            "wannier_function_centers": Artifact(Dict[str, Path]),
            "failed_confs": Artifact(Path),
        })

    @OP.exec_sign_check # type: ignore
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        mlwf_setting: dict = op_in["mlwf_setting"]
        conf_fmt = op_in["conf_fmt"]
        conf_sys = read_conf(op_in["confs"], conf_fmt)
        backward: List[Path] = op_in["backward"]

        total_wfc, final_confs, failed_confs = self.collect_wfc(mlwf_setting, conf_sys, backward)
        wfc_path: Dict[str, Path] = {}
        for key, wfc in total_wfc.items():
            wfc_path[key] = Path(f"wfc_{key}.raw")
            np.savetxt(wfc_path[key], wfc)
        final_confs_path, final_conf_fmt = dump_to_fmt(
            "final_confs", final_confs, fmt = "deepmd/npy", in_fmt = conf_fmt, set_size = 5000
        )
        failed_confs_path, _ = dump_to_fmt(
            "failed_confs", failed_confs, fmt = "deepmd/npy", set_size = 5000
        )
        return OPIO({
            "final_confs": final_confs_path,
            "final_conf_fmt": final_conf_fmt,
            "wannier_function_centers": wfc_path,
            "failed_confs": failed_confs_path
        })
    
    def collect_wfc(self, mlwf_setting: dict, conf_sys: dpdata.System, backward: List[Path]):
        assert conf_sys.get_nframes() == len(backward)
        self.init_params(mlwf_setting, conf_sys, backward[0])
        total_wfc: Dict[str, np.ndarray] = {}
        failed_frames: List[int] = []
        success_frames: List[int] = []
        def update_wfc(wfc_frame: Dict[str, np.ndarray], frame: int):
            for key, wfc_arr in wfc_frame.items():
                if key not in total_wfc:
                    total_wfc[key] = np.zeros((conf_sys.get_nframes(), wfc_arr.size), dtype = float)
                total_wfc[key][frame] = wfc_arr.flatten()
        for frame, p in enumerate(backward):
            with set_directory(p):
                try:
                    wfc_frame = self.get_one_frame()
                    update_wfc(wfc_frame, frame)
                    success_frames.append(frame)
                except Exception as e:
                    print(f"[ERROR] Failed to read results of frame {frame:06d}: {e}")
                    failed_frames.append(frame)
        assert len(success_frames) > 0, "All frames failed!"
        final_confs = conf_sys.sub_system(success_frames)
        for key in total_wfc:
            total_wfc[key] = total_wfc[key][success_frames]
        if len(failed_frames) > 0:
            failed_confs = conf_sys.sub_system(failed_frames)
        else:
            failed_confs = None
        return total_wfc, final_confs, failed_confs


    @abc.abstractmethod
    def get_one_frame(self) -> Dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def init_params(self, mlwf_setting: dict, conf_sys: dpdata.System, example_file: Path):
        pass
