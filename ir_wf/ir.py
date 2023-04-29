from typing import Dict, List, Tuple, Union
from pathlib import Path
import dpdata, numpy as np
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter,
)
from dflow.utils import (
    set_directory
)
from ir_wf.cal_ir import calculate_corr, calculate_ir

class IR(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "ir_setting": BigParameter(Dict),
            "total_dipole": Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "ir": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        ir_setting = op_in["ir_setting"]
        dt = ir_setting["dt"]
        width = ir_setting["width"]
        temperature = ir_setting["temperature"]
        window = ir_setting["window"]

        total_dipole = np.load(op_in["total_dipole"])
        v_dipole = (total_dipole[2:] - total_dipole[:-2]) / (2 * dt)
        v_dipole -= np.mean(v_dipole, axis = 0, keepdims = True)
        
        corr = calculate_corr(v_dipole, v_dipole, window)

        ir = calculate_ir(corr, width = width, dt = dt, temperature = temperature)
        ir_path = Path("ir.raw")
        np.savetxt(ir_path, ir, fmt = "%15.8f")
        return OPIO({
            "ir": ir_path,
        })
    