import numpy as np

def box_shift(dx: np.ndarray, box: np.ndarray):
    nl = np.floor(dx / (box / 2))
    return dx - (nl + (nl + 2) % 2) * box / 2

def k_nearest(coords_A: np.ndarray, coords_B: np.ndarray, box: np.ndarray, k: int):
    """
        For each point in coords_A, choose the k-nearest points (in the box) among coords_B, and return the index.
        The distance is calculated in the sense of PBC.

        Parameters
        -------------
            coords_A (..., num_A, d): the coordinates of the central points. The size of the last axis is the dimension.
            coords_B (..., num_B, d): the coordinates of the points to be selected.
            box      (..., d): the PBC box. box[..., i] is the length of period along x_i.
            k: int, the number of the points selected from coords_B.

        Return
        -------------
            index (..., num_A, k): the index of the k-nearest points in coords_B.
    """
    distance = np.linalg.norm(
        box_shift(
            coords_A[..., np.newaxis, :] - coords_B[..., np.newaxis, :, :], 
            box[..., np.newaxis, np.newaxis, :]
        ), 
        ord = 2, axis = -1
    )
    return np.argsort(distance, axis = -1)[..., :k]

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

if __name__ == '__main__':
    with open('./water_centres.xyz', 'r') as f:
        wfc = np.loadtxt(f, dtype = float, skiprows = 2, usecols = [1, 2, 3], max_rows = 256)
        coords_O = np.loadtxt(f, dtype = float, usecols = [1, 2, 3], max_rows = 64)
    box = np.ones((3, ), dtype = float) * 12.136090
    wc = cal_wc_h2o(wfc, coords_O, box)
    np.savetxt('wc.raw', wc, fmt = '%f')