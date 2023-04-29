from typing import Dict, List, Tuple, Union
from pathlib import Path
import dpdata, numpy as np, abc
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

class CollectWFC(OP, abc.ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "input_setting": BigParameter(dict),
            "confs": Artifact(Path),
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
        self.confs = dpdata.System(op_in["confs"], fmt='deepmd/raw', type_map = ['O', 'H'])
        backward: List[Path] = op_in["backward"]

        wfc = self.collect_wfc(input_setting, backward)
        return OPIO({
            "wannier_function_centers": wfc
        })
    
    def collect_wfc(self, input_setting: dict, backward: List[Path]) -> Dict[str, Path]:
        if not backward:
            return np.array([])
        self.init_params(input_setting, backward)
        wfc = np.zeros((len(backward), self.num_wann * 3), dtype = float)
        for frame, p in enumerate(backward):
            with set_directory(p):
                wfc[frame] = self.get_one_frame(frame).flatten()
        wfc_path = Path("wfc.raw")
        np.savetxt(wfc_path, wfc, fmt = "%15.8f")
        return {
            "ori": wfc_path
        }

    @abc.abstractmethod
    def get_one_frame(self, frame: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def init_params(self, input_setting: dict, backward: List[Path]):
        try:
            self.num_wann = input_setting["num_wann"]
        except KeyError:
            self.num_wann = self.get_num_wann(backward[0])

    @abc.abstractmethod
    def get_num_wann(self, file_path: Path) -> int:
        pass