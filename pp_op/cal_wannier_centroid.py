import numpy as np

def box_shift(dx: np.ndarray, box: np.ndarray):
    nl = np.floor(dx / (box / 2))
    return dx - (nl + (nl + 2) % 2) * box / 2

def k_nearest(coords_A: np.ndarray, coords_B: np.ndarray, box: np.ndarray, k: int):
    distance = np.linalg.norm(
        box_shift(
            coords_A[..., np.newaxis, :] - coords_B[..., np.newaxis, :, :], 
            box[..., np.newaxis, np.newaxis, :]
        ), 
        ord = 2, axis = -1
    )
    return np.argsort(distance, axis = -1)[..., :k]

def cal_wc_h2o(wfc: np.ndarray, coords_O: np.ndarray, box: np.ndarray):
    idx = k_nearest(coords_O, wfc, box, k = 4)
    wc = np.take_along_axis(wfc[..., np.newaxis, :, :], idx[..., np.newaxis], axis = -2)
    return np.mean(box_shift(wc - coords_O[..., np.newaxis, :], box), axis = -2)

if __name__ == '__main__':
    with open('./water_centres.xyz', 'r') as f:
        wfc = np.loadtxt(f, dtype = float, skiprows = 2, usecols = [1, 2, 3], max_rows = 256)
        coords_O = np.loadtxt(f, dtype = float, usecols = [1, 2, 3], max_rows = 64)
    box = np.ones((3, ), dtype = float) * 12.136090
    wc = cal_wc_h2o(wfc, coords_O, box)
    np.savetxt('wc.raw', wc, fmt = '%f')