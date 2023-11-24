from spectra_flow.dp.infer import model_eval
from pathlib import Path
from typing import Union
import dpdata, numpy as np, importlib
from spectra_flow.post.cal_dipole import calculate_dipole_h2o
from spectra_flow.utils import do_pbc

def eval_deep_dipole(model_path: Union[Path, str], sys: dpdata.System):
    DeepDipole = importlib.import_module("deepmd.infer").DeepDipole
    deep_wannier = DeepDipole(model_path)
    predicted_wc = model_eval(deep_wannier, sys, set_size = 128)
    predicted_wc /= 40.0    # amplif
    return predicted_wc

def calc_dipole_h2o(sys: dpdata.System, wc: np.ndarray):
    mask_O = sys["atom_types"] == 0
    coords = do_pbc(sys["coords"], sys["cells"]) # type: ignore
    nframes = sys.get_nframes()
    wc = wc.reshape(nframes, -1, 3)
    return calculate_dipole_h2o(
        coords[:, mask_O], coords[:, ~mask_O], sys["cells"], wc, r_bond = 1.2 # type: ignore
    ).reshape(nframes, -1)

if __name__ == "__main__":
    sys = dpdata.System("sys.lmp", fmt = "lammps/lmp", type_map = ["O", "H"])
    wc = eval_deep_dipole("dipole.pb", sys)
    dipole = calc_dipole_h2o(sys, wc)
    print(dipole)