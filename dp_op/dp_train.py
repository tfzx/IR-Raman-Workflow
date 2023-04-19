from typing import Dict, List, Tuple, Union
from pathlib import Path
import dpdata, numpy as np, json
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
from mlwf_op.utils import filter_confs

class DPTrain(OP):
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
            "dp_train_inputs": BigParameter(Dict),
            "tensor": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "frozen_model": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        confs = dpdata.System(op_in["confs"], fmt='deepmd/raw', type_map = ['O', 'H'])
        tensor = np.loadtxt(op_in["tensor"], dtype = float)
        train_inputs = op_in["dp_train_inputs"]
        train_dir = self.prepare_train(train_inputs, confs, tensor)
        model = self.run_train(train_dir)
        return OPIO({
            "frozen_model": model
        })
    
    def prepare_train(self, train_inputs: Dict, confs: dpdata.System, tensor: np.ndarray, set_size: int = 5000):
        data_dir = Path("data")
        train_dir = Path("train")
        data_dir.mkdir()
        train_dir.mkdir()
        confs, tensor = filter_confs(confs, tensor)
        confs.to("deepmd/npy", data_dir, set_size = set_size)
        nframes = confs.get_nframes()
        start_i = 0
        idx = 0
        while start_i < nframes:
            end_i = min(start_i + set_size, nframes)
            np.save(data_dir / Path(f"set.{idx:03d}") / Path(f"atomic_{self.full_name[self.dp_type]}.npy"), tensor[start_i:end_i])
        train_inputs["training"].update({
            "systems": [str(data_dir.absolute())],
            "set_prefix": "set"
        })
        with open(train_dir / Path("input.json"), "w+") as f:
            json.dump(train_inputs, fp = f)
        return train_dir

    def run_train(self, train_dir: Path):
        with set_directory(train_dir):
            run_command("dp train input.json")
            run_command(f"dp freeze -o {self.dp_type}.pb")
            model = Path(f"{self.dp_type}.pb").absolute()
        return model