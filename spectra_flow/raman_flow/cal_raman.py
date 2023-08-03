from typing import Optional
import numpy as np
from spectra_flow.utils import (
    calculate_corr,
    apply_gussian_filter,
    FT
)

def calculate_corr_polar(polar: np.ndarray, window: int):
    polar_iso = np.mean(polar.diagonal(offset = 0, axis1 = 1, axis2 = 2), axis = 1)

    diag = np.zeros((polar_iso.shape[0], 3, 3), dtype = float)
    diag[:, 0, 0] = polar_iso
    diag[:, 1, 1] = polar_iso
    diag[:, 2, 2] = polar_iso
    polar_aniso = polar - diag # type: ignore

    polar_iso -= np.mean(polar_iso, axis = 0, keepdims = True)
    polar_aniso -= np.mean(polar_aniso, axis = 0, keepdims = True)

    corr_iso = np.sum(calculate_corr(polar_iso, polar_iso, window), axis = -1)
    polar_aniso = polar_aniso.reshape(-1, 9)
    corr_aniso = np.sum(calculate_corr(polar_aniso, polar_aniso, window), axis = -1)
    corr_aniso *= 2 / 15
    return corr_iso, corr_aniso

def calculate_raman(corr: np.ndarray, width: float, dt_ps: float, temperature: float, M: Optional[int] = None):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt_ps        # ps
    print('nmax      =', nmax)
    print('dt   (ps) =', dt_ps)
    print('tmax (ps) =', tmax)
    print("width     = ", width)
    width = width * tmax / 100.0 * 3.0
    C = apply_gussian_filter(corr, width)
    freq_ps, CHAT = FT(dt_ps, C, M)
    d_omega, CHAT = _change_unit(freq_ps, CHAT, temperature)
    return np.stack([np.arange(CHAT.shape[0]) * d_omega, CHAT], axis = 1)

def _change_unit(freq_ps, CHAT: np.ndarray, temperature: float):
    cc = 2.99792458e8;                  # m/s
    kB = 1.38064852*1.0e-23             # J/K
    h = 6.62607015e-34                  # J*s
    h_bar = h / (2 * np.pi)
    beta = 1.0 / (kB * temperature);    # J^-1
    freq = 2 * np.pi * freq_ps * 1e12   # s^-1
    CHAT = CHAT * 1e4 * (1 - np.exp(-beta * h_bar * freq * np.arange(CHAT.shape[0])))
    d_omega = 1e10 * freq_ps / cc       # cm^-1
    return d_omega, CHAT