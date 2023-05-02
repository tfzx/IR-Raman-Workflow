import numpy as np

def box_shift(dx: np.ndarray, box: np.ndarray):
    nl = np.floor(dx / (box / 2))
    return dx - (nl + (nl + 2) % 2) * box / 2

def _fix_coords(coords_sel: np.ndarray, coords_oth: np.ndarray, box: np.ndarray, r_bond: float, mask_sel: np.ndarray = None):
    coords_sel = coords_sel[..., np.newaxis, :]
    delta = box_shift(coords_oth[..., np.newaxis, :, :] - coords_sel, box[..., np.newaxis, np.newaxis, :])
    mask = np.linalg.norm(delta, 2, axis = -1, keepdims = True) < r_bond
    if mask_sel is not None:
        mask &= mask_sel[..., np.newaxis, np.newaxis]
    delta += coords_sel
    ans = np.sum(np.where(mask, delta, 0), axis = -3)
    num_nb = np.sum(mask, axis = -3)
    ans = np.where(num_nb > 0, ans, coords_oth)
    num_nb[num_nb == 0] = 1
    coords_oth[...] = ans / num_nb

def fix_coords(coords_sel: np.ndarray, coords_oth: np.ndarray, box: np.ndarray, r_bond: float):
    c = np.concatenate([coords_sel, coords_oth, box[..., np.newaxis, :]], axis = -2).reshape(coords_sel.shape[0], -1)
    num_sel = coords_sel.shape[-2]
    def fix(arr):
        arr = arr.reshape(-1, 3)
        c_sel = arr[:num_sel]
        c_oth = arr[num_sel:-1]
        b = arr[[-1]]
        mask_sel = np.abs(box_shift(c_sel, b)).min(axis = -1) < 2 * r_bond
        mask_oth = np.abs(box_shift(c_oth, b)).min(axis = -1) < r_bond
        c_sel_small = c_sel[mask_sel, :]
        c_oth_small = c_oth[mask_oth, :]
        _fix_coords(c_sel_small, c_oth_small, b, r_bond)
        c_oth[mask_oth, :] = c_oth_small[np.arange(c_oth_small.shape[0]), :]
        return True
    np.apply_along_axis(fix, axis = -1, arr = c)
    return c.reshape(coords_sel.shape[0], -1, 3)[:, num_sel:-1, :]

def calculate_dipole_h2o(coords_sel: np.ndarray, coords_oth: np.ndarray, box: np.ndarray, wannier: np.ndarray, r_bond = 1.2) -> np.ndarray:
    coords_oth = fix_coords(coords_sel, coords_oth, box, r_bond)
    total_dipole = np.sum(coords_oth, axis = 1) - np.sum(coords_sel, axis = 1) * 2. - np.sum(wannier, axis = 1) * 8.
    vol = np.prod(box, axis = -1, keepdims = True)
    return total_dipole / np.sqrt(vol) * np.sqrt(0.52917721067)

if __name__ == '__main__':
    type_O, type_H = 0, 1
    amplif = 10.0
    a0 = 0.52917721067
    debye2ea=0.20819434
    dt = 0.0003

    coords = np.loadtxt('coord.raw')[:1]
    cells = np.loadtxt('box.raw')[:1]
    types = np.loadtxt('type.raw', dtype = int)
    wannier = np.loadtxt('dipole.raw')[:1]

    nFrames = coords.shape[0]
    nAtoms = int(coords.shape[1] / 3)
    coords = coords.reshape(nFrames, -1, 3)
    wannier = wannier.reshape(nFrames, -1, 3) / (4 * amplif)
    box = cells.reshape(nFrames, 3, 3).diagonal(offset = 0, axis1 = 1, axis2 = 2)

    coords_O = coords[:, types == type_O, :]
    coords_H = coords[:, types == type_H, :]
    total_dipole = calculate_dipole_h2o(coords_O, coords_H, box, wannier)
    print(total_dipole)