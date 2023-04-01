from typing import List
from mlwf_op.collect_wfc_op import CollectWFC
import numpy as np
from pathlib import Path

class CollectCPWF(CollectWFC):
    a0 = 0.5291772083
    def init_params(self, input_setting: dict, backward: List[Path]):
        self.prefix = input_setting["dft_params"]["qe_params"]["control"]["prefix"]
        return super().init_params(input_setting, backward)

    def get_num_wann(self, file_path: Path) -> int:
        start = False
        num_wann = 0
        with open(file_path / f"{self.prefix}.wfc", "r") as fp:
            for line in fp.readlines():
                line = line.strip()
                if not line:
                    break
                if len(line.split()) > 3:
                    if start:
                        break
                    else:
                        start = True
                elif start:
                    num_wann += 1
        return num_wann

    def get_one_frame(self, frame: int) -> np.ndarray:
        with open(f"{self.prefix}.wfc", "r") as fp:
            while fp.readline().strip():
                wann = np.loadtxt(fp, dtype = float, max_rows = self.num_wann)
        return wann * self.a0