from typing import Dict, List, Tuple, Union
from pathlib import Path
import dpdata, numpy as np, json, abc
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter,
)
from dflow.utils import (
    set_directory,
    run_command
)
from spectra_flow.utils import filter_confs, read_conf

class DPTrain(OP, abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dp_type = "dipole" # "dipole" or "polar"
        self.full_name = {
            "dipole": "dipole",
            "polar": "polarizability"
        }
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "confs": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "dp_setting": BigParameter(Dict),
            "label": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "frozen_model": Artifact(Path),
            "lcurve": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        confs = read_conf(op_in["confs"], op_in["conf_fmt"])
        label = np.loadtxt(op_in["label"], dtype = float)
        dp_setting: dict = op_in["dp_setting"]
        train_inputs: dict = dp_setting["train_inputs"]
        label = self.preprocess(label, dp_setting)
        train_dir = self.prepare_train(train_inputs, confs, label)
        model, lcurve = self.run_train(train_dir)
        model.symlink_to(train_dir / model)
        lcurve.symlink_to(train_dir / lcurve)
        return OPIO({
            "frozen_model": model,
            "lcurve": lcurve
        })

    @abc.abstractmethod
    def preprocess(self, label: np.ndarray, dp_setting: dict) -> np.ndarray:
        return label

    def prepare_train(self, train_inputs: Dict[str, dict], confs: dpdata.System, label: np.ndarray, set_size: int = 5000):
        data_dir = Path("data")
        train_dir = Path("train")
        data_dir.mkdir()
        train_dir.mkdir()
        confs, label = filter_confs(confs, label)
        confs.to("deepmd/npy", data_dir, set_size = set_size)
        nframes = confs.get_nframes()
        start_i = 0
        idx = 0
        while start_i < nframes:
            end_i = min(start_i + set_size, nframes)
            np.save(data_dir / Path(f"set.{idx:03d}") / Path(f"atomic_{self.full_name[self.dp_type]}.npy"), label[start_i:end_i])
            start_i += set_size
            idx += 1
        train_inputs["training"].update({
            "systems": [str(data_dir.absolute())],
            "set_prefix": "set"
        })
        self.lcurve_name = train_inputs["training"].setdefault("disp_file", "lcurve.out")
        with open(train_dir / Path("input.json"), "w+") as f:
            json.dump(train_inputs, fp = f)
        return train_dir

    def run_train(self, train_dir: Path):
        with set_directory(train_dir):
            run_command("export OMP_NUM_THREADS=20 && dp train input.json", try_bash = True, print_oe = True)
            run_command(f"dp freeze -o {self.dp_type}.pb", try_bash = True, print_oe = True)
            model = Path(f"{self.dp_type}.pb")
            lcurve = Path(self.lcurve_name)
        return model, lcurve

class DWannTrain(DPTrain):
    def __init__(self) -> None:
        super().__init__()
        self.dp_type = "dipole"
    
    def preprocess(self, label: np.ndarray, dp_setting: dict) -> np.ndarray:
        amplif = dp_setting.get("amplif", 1.0)
        return label * amplif

class DPolarTrain(DPTrain):
    def __init__(self) -> None:
        super().__init__()
        self.dp_type = "polar"
    
    def preprocess(self, label: np.ndarray, dp_setting: dict) -> np.ndarray:
        return label