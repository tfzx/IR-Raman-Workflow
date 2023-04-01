from typing import List
from mlwf_op.collect_wfc_op import CollectWFC
import numpy as np
from pathlib import Path

class CollectWann(CollectWFC):
    def get_one_frame(self, frame: int) -> np.ndarray:
        return np.loadtxt(f'{self.name}_centres.xyz', dtype = float, skiprows = 2, usecols = [1, 2, 3], max_rows = self.num_wann)
    
    def init_params(self, input_setting: dict, backward: List[Path]):
        self.name = input_setting["name"]
        return super().init_params(input_setting, backward)

    def get_num_wann(self, file_path: Path) -> int:
        num_wann = int(np.loadtxt(file_path / f'{self.name}_centres.xyz', dtype = int, max_rows = 1)) - self.confs.get_natoms()
        return num_wann