from typing import Dict
import numpy as np
import dpdata

def model_eval(model, smp_sys: dpdata.System, set_size: int = 128) -> np.ndarray:
    coord = smp_sys["coords"]
    cell = smp_sys["cells"]
    atype = smp_sys["atom_types"]
    nframes = coord.shape[0] # type: ignore
    batch = 0
    out_all = []
    while batch < nframes:
        print("-----------------------------------", "current batch", batch, "-----------------------------------")
        batch_n = min(batch + set_size, nframes)
        out = model.eval(coord[batch:batch_n], cell[batch:batch_n], atype)
        out_all.append(out)
        batch = batch_n
    return np.concatenate(out_all, axis = 0).reshape([nframes, -1])