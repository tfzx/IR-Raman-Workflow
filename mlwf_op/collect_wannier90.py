from mlwf_op.collect_wfc_op import CollectWFC
import numpy as np

class CollectWann(CollectWFC):
    def get_one_frame(self, frame: int, num_wann: int) -> np.ndarray:
        return np.loadtxt(f'{self.name}_centres.xyz', dtype = float, skiprows = 2, usecols = [1, 2, 3], max_rows = num_wann)