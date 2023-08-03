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
from spectra_flow.raman_flow.cal_raman import calculate_corr_polar, calculate_raman

class CalRaman(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "global": BigParameter(Dict),
            "total_polar": Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "raman_iso": Artifact(Path),
            "raman_aniso": Artifact(Path)
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

        total_polar = np.load(op_in["total_polar"]).reshape(-1, 3, 3)
        
        corr_iso, corr_aniso = calculate_corr_polar(total_polar, window)
        raman_iso = calculate_raman(corr_iso, width = width, dt_ps = dt, temperature = temperature, M = M)
        raman_aniso = calculate_raman(corr_aniso, width = width, dt_ps = dt, temperature = temperature, M = M)

        raman_iso_path = Path("raman_iso.raw")
        raman_aniso_path = Path("raman_aniso.raw")
        np.savetxt(raman_iso_path, raman_iso, fmt = "%15.8f")
        np.savetxt(raman_aniso_path, raman_aniso, fmt = "%15.8f")
        return OPIO({
            "raman_iso": raman_iso_path,
            "raman_aniso": raman_aniso_path
        })
    