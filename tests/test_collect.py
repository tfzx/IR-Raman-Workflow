from spectra_flow.mlwf.mlwf_ops import CollectWFC
from pathlib import Path
import numpy as np

def test_collect():
    collect_op = CollectWFC()
    op_in = {
        "name": "water",
        "confs": Path("./data"),
        "backward": [Path("./back")]
    }
    op_out = collect_op.execute(op_in)
    print(np.loadtxt(op_out["wannier_function_centers"], dtype = float))

if __name__ == "__main__":
    test_collect()