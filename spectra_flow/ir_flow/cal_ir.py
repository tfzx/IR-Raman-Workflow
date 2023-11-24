from typing import Optional
import numpy as np
from spectra_flow.utils import (
    calculate_corr,
    apply_gussian_filter,
    apply_lorenz_filter,
    FT
)

def calculate_corr_vdipole(dipole: np.ndarray, dt_ps: float, window: int):
    v_dipole = (dipole[1:] - dipole[:-1]) / dt_ps # type: ignore
    v_dipole -= np.mean(v_dipole, axis = 0, keepdims = True)
    corr = np.sum(calculate_corr(v_dipole, v_dipole, window), axis = -1)
    return corr

def calculate_ir(corr: np.ndarray, width: float, dt_ps: float, temperature: float, 
                 M: Optional[int] = None, filter_type: str = "gaussian"):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt_ps
    filter_type = filter_type.lower().strip()
    print("nmax         =", nmax)
    print("dt   (ps)    =", dt_ps)
    print("tmax (ps)    =", tmax)
    print("Filter type  =", filter_type)
    print("Smooth width =", width)
    if filter_type == "gaussian":
        width = width * tmax / 100.0 * 3
        C = apply_gussian_filter(corr, width)
    elif filter_type == "lorenz":
        C = apply_lorenz_filter(corr, width, dt_ps)
    else:
        raise NotImplementedError(f"Unknown filter type: {filter_type}!")
    freq_ps, CHAT = FT(dt_ps, C, M)
    d_omega, CHAT = _change_unit(freq_ps, CHAT, temperature)
    return np.stack([np.arange(CHAT.shape[0]) * d_omega, CHAT], axis = 1)

def _change_unit(freq_ps, CHAT: np.ndarray, temperature: float):
    a0 = 0.52917721067e-10  # m
    cc = 2.99792458e8;      # m/s
    kB = 1.38064852*1.0e-23 # J/K
    h = 6.62607015e-34      # J*s
    beta = 1.0 / (kB * temperature); 
	# 1 Debye = 0.20819434 e*Angstrom
	# 1 e = 1.602*1.0e-19 C
	# change unit to C*m for M(0)
    unit_basic = 1.602176565 * 1.0e-19 * a0
	# change unit to ps for dM(0)/dt
    unitt = unit_basic / 1
	# because dot(M(0))*dot(M(t)) change unit to C^2 * m^2 / ps^2
    unit2 = unitt**2
    epsilon0 = 8.8541878e-12 # F/m = C^2 / (J * m)
    unit_all = beta / (3.0 * cc * a0 ** 3) / (2 * epsilon0) * unit2
    unit_all = unit_all * 1.0e12 * 1.0e-2; # ps to s, m-1 to cm-1
    CHAT *= unit_all
    d_omega = freq_ps / cc     # Wavenumber
    d_omega *= 1e10         # cm^-1
    return d_omega, CHAT
