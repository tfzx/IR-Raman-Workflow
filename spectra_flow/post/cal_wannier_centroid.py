import numpy as np
import dpdata
from spectra_flow.utils import k_nearest

def box_shift(dx: np.ndarray, box: np.ndarray):
    nl = np.floor(dx / (box / 2))
    return dx - (nl + (nl + 2) % 2) * box / 2

def cal_wc_h2o(wfc: np.ndarray, coords_O: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
        Calculate the wannier centroids for system of H2O.

        Parameters
        -------------
            wfc      (..., num_wann, 3): the coordinates of the wannier function centers. 
            coords_O (..., num_O, 3): the coordinates of the O atoms. It should be num_wann = 4 * num_O.
            box      (..., 3): the PBC box. box[..., i] is the length of period along x_i.

        Return
        -------------
            wannier centroids (..., num_O, 3): the wannier centroid relative to each O atoms.
    """
    idx = k_nearest(coords_O, wfc, box, k = 4)
    wc = np.take_along_axis(wfc[..., np.newaxis, :, :], idx[..., np.newaxis], axis = -2)
    return np.mean(box_shift(wc - coords_O[..., np.newaxis, :], box[..., np.newaxis, np.newaxis, :]), axis = -2)

def cal_wc(confs: dpdata.System, wfc: np.ndarray) -> np.ndarray:
        return cal_wc_h2o(
            wfc.reshape(confs.get_nframes(), -1, 3), 
            confs["coords"][:, confs["atom_types"] == 0], 
            np.diagonal(confs["cells"], axis1 = -2, axis2 = -1)
        ).reshape(confs.get_nframes(), -1)
