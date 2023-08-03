from typing import Dict
from pathlib import Path
import dpdata, numpy as np
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter,
)
from spectra_flow.ir_flow.cal_ir import calculate_corr_vdipole, calculate_ir

class CalIR(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "global": BigParameter(Dict),
            "total_dipole": Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "ir": Artifact(Path),
        })

    @OP.exec_sign_check # type: ignore
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        global_config = op_in["global"]
        dt = global_config["dt"]
        width = global_config["width"]
        temperature = global_config["temperature"]
        window = global_config["window"]
        M = global_config.get("num_omega", None)
        if M is not None:
            M = max(M, window)

        total_dipole = np.load(op_in["total_dipole"]).reshape(-1, 3)
        
        corr = calculate_corr_vdipole(total_dipole, dt_ps = dt, window = window)
        ir = calculate_ir(corr, width = width, dt_ps = dt, temperature = temperature, M = M)

        ir_path = Path("ir.raw")
        np.savetxt(ir_path, ir, fmt = "%15.8f")
        return OPIO({
            "ir": ir_path,
        })
    