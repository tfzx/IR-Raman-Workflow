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
from deepmd.infer import DeepDipole

class DWannPredict(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "sampled_system": Artifact(Path),
            "frozen_model": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "predicted_wc": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        smp_sys = dpdata.System(op_in["sampled_system"], fmt='deepmd/raw', type_map = ['O', 'H'])
        deep_wannier = DeepDipole(op_in["frozen_model"])
        predicted_wc = self._predict(deep_wannier, smp_sys)
        wc_path = Path("predicted_wc.raw")
        np.savetxt(wc_path, predicted_wc, fmt = "%15.8f")
        return OPIO({
            "predicted_wc": wc_path
        })
        
    def _predict(self, model, smp_sys: Dict[str, np.ndarray], set_size: int = 128):
        coord = smp_sys["coords"]
        cell = smp_sys["cells"]
        atype = smp_sys["atom_types"]
        nframes = coord.shape[0]
        batch = 0
        out_all = []
        while batch < nframes:
            print("-------------------------------------", "current batch", batch, "-----------------------------------")
            batch_n = min(batch + set_size, nframes)
            out = model.eval(coord[batch:batch_n], cell[batch:batch_n], atype)
            out_all.append(out)
            batch = batch_n
        return np.concatenate(out_all, axis = 0).reshape([nframes, -1])