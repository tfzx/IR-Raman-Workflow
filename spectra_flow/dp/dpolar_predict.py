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
from spectra_flow.dp.infer import model_eval

class DPolarPredict(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "dp_setting": BigParameter(Dict),
            "sampled_system": Artifact(Path),
            "frozen_model": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "predicted_polar": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        sys_path: Path = op_in["sampled_system"]
        dp_setting = op_in["dp_setting"]
        if "dump_fmt" in dp_setting:
            dump_fmt: str = dp_setting["dump_fmt"]
            dump_fmt = dump_fmt.strip()
            head = dump_fmt.split("/")[0]
            if head == "numpy":
                smp_sys = np.load(sys_path)
            else:
                smp_sys = dpdata.System(op_in["sampled_system"], fmt = dump_fmt)
        else:
            if sys_path.is_file():
                smp_sys = np.load(sys_path)
            else:
                smp_sys = dpdata.System(op_in["sampled_system"])
            
        from deepmd.infer import DeepPolar
        deep_polar = DeepPolar(op_in["frozen_model"])
        predicted_polar = model_eval(deep_polar, smp_sys)
        polar_path = Path("predicted_polar.raw")
        np.save(polar_path, predicted_polar)
        return OPIO({
            "predicted_polar": polar_path
        })
    