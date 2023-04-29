from typing import Tuple
import numpy as np

'''
def calculate_corr(A: np.ndarray, B: np.ndarray, window: int, n: int):
    v1 = A[:n][::-1]
    v2 = B[:n + window]
    corr = np.convolve(v1[:, 0], v2[:, 0], 'valid') \
            + np.convolve(v1[:, 1], v2[:, 1], 'valid') \
            + np.convolve(v1[:, 2], v2[:, 2], 'valid')
    corr /= n
    return corr
'''

def calculate_corr(A: np.ndarray, B: np.ndarray, window: int, n: int = None):
    if A.ndim == 1 or B.ndim == 1:
        A = A.reshape(-1, 1)
        B = B.reshape(-1, 1)
    if not n:
        n = min(A.shape[0], B.shape[0])
    assert n <= min(A.shape[0], B.shape[0]), "The number of steps is too large!"
    v1 = np.concatenate([A[:n][::-1], np.zeros([window, A.shape[1]], dtype = np.float32)], axis = 0)
    v2 = B[:n + window]
    corr = np.fft.ifft(np.fft.fft(v1, axis = 0) * np.fft.fft(v2, axis = 0), axis = 0).real
    corr = np.sum(corr[n - 1:n + window] / n, axis = -1)
    return corr

def calculate_ir(corr: np.ndarray, width: int, dt: float, temperature: float):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt
    # dom = 2. * np.pi / tmax
    print('nmax =', nmax)
    print('dt   =', dt)
    print('tmax =', tmax)
    print("width = ", width)
    width = width * tmax / 100.0 * 3.0
    C = apply_gussian_filter(corr, width)
    CHAT = FT(dt, C)
    a0 = 0.52917721067e-10  # m
    cc = 2.99792458e8;      # m/s
    kB = 1.38064852*1.0e-23 # J/K
    h = 6.62607015e-34      # J*s
    h_bar = h / (2 * np.pi)
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
    d_omega = 1e10 / (tmax * cc)
    return np.stack([np.arange(CHAT.shape[0]) * d_omega, CHAT], axis = 1)

def apply_gussian_filter(corr: np.ndarray, width: int):
    nmax = corr.shape[0] - 1
    return corr * np.exp(-.5 * (0.5 * width * np.arange(nmax + 1) / nmax)**2)

def FILONC(DT: float, DOM: float, C: np.ndarray) -> np.ndarray:
    NMAX = C.shape[0] - 1
    if NMAX % 2 != 0:
        return
    TMAX = NMAX * DT
    NU = np.arange(NMAX + 1)
    OMEGA = NU * DOM
    THETA = OMEGA * DT
    SINTH = np.sin(THETA)
    COSTH = np.cos(THETA)
    SINSQ = np.square(SINTH)
    COSSQ = np.square(COSTH)
    THSQ  = np.square(THETA)
    THCUB = THSQ * THETA
    ALPHA = 1. * ( THSQ + THETA * SINTH * COSTH - 2. * SINSQ )
    BETA  = 2. * ( THETA * ( 1. + COSSQ ) - 2. * SINTH * COSTH )
    GAMMA = 4. * ( SINTH - THETA * COSTH )
    ALPHA[0] = 0.
    BETA[0] = 2. / 3.
    GAMMA[0] = 4. / 3.
    ALPHA[1:] /= THCUB[1:]
    BETA[1:] /= THCUB[1:]
    GAMMA[1:] /= THCUB[1:]
    def CAL_C0(theta: np.ndarray, args: Tuple[np.ndarray, np.ndarray]):
        a, b = args
        return np.dot(a, np.cos(theta * b))
    CE = np.apply_along_axis(CAL_C0, axis = 1, arr = THETA[:, np.newaxis], args = (C[::2], np.arange(0, NMAX + 1, 2)))
    CE -= 0.5 * (C[0] + C[NMAX] * np.cos(OMEGA * TMAX))
    CO = np.apply_along_axis(CAL_C0, axis = 1, arr = THETA[:, np.newaxis], args = (C[1::2], np.arange(1, NMAX, 2)))
    CHAT = 2.0 * ( ALPHA * C[NMAX] * np.sin ( OMEGA * TMAX ) + BETA * CE + GAMMA * CO ) * DT
    return CHAT

def FT(DT: float, C: np.ndarray) -> np.ndarray:
    NMAX = C.shape[0] - 1
    assert NMAX % 2 == 0, 'NMAX is not even!'
    DTH = 2 * np.pi / NMAX
    NU = np.arange(NMAX + 1)
    THETA = NU * DTH
    SINTH = np.sin(THETA)
    COSTH = np.cos(THETA)
    COSSQ = np.square(COSTH)
    THSQ  = np.square(THETA)
    THCUB = THSQ * THETA
    BETA  = 2. * ( THETA * ( 1. + COSSQ ) - 2. * SINTH * COSTH )
    GAMMA = 4. * ( SINTH - THETA * COSTH )
    BETA[0] = 2. / 3.
    GAMMA[0] = 4. / 3.
    BETA[1:] /= THCUB[1:]
    GAMMA[1:] /= THCUB[1:]
    CE = np.fft.fft(C[:-1:2]).real + 0.5 * (C[NMAX] - C[0])
    CO = (np.fft.fft(C[1::2]) * np.exp(-THETA[:int(NMAX / 2)] * 1j)).real
    CE = np.concatenate([CE, CE, CE[0:1]])
    CO = np.concatenate([CO, -CO, CO[0:1]])
    CHAT = 2.0 * (BETA * CE + GAMMA * CO) * DT
    return CHAT

def numerical_diff(y: np.ndarray, h: float):
    g = (y[2:] - y[:-2]) / (2 * h)
    g3, g4   = diff_8(g[:8])    # approx g[3], g[4]
    gn5, gn4 = diff_8(g[-8:])   # approx g[-4], g[-5]
    g[4] += (gn5 - g3) / 6.
    g[-5] += (g4 - gn4) / 6.
    g = g[4:-4]
    v = np.zeros((g.shape[0], 1))
    v[0] = -2; v[1] = 1; v[-1] = 1
    return np.fft.ifft(np.fft.fft(g, axis = 0) / (1 + np.fft.fft(v, axis = 0) / 6), axis = 0).real

def diff_8(g):
    w = np.array([336./2911., -1344./2911., 5040./2911., -1350./2911., 360./2911., -90./2911.])
    b = g[1:7].copy()
    b[0] -= g[0] / 6.
    b[-1] -= g[7] / 6.
    g_3 = np.dot(w, b)   # approx g[3]
    g_4 = np.dot(w[::-1], b)
    return g_3, g_4

if __name__ == '__main__':
    type_O, type_H = 0, 1
    amplif = 10.0
    a0 = 0.52917721067
    debye2ea=0.20819434
    dt = 0.0003

    total_dipole = np.load('2deepwannier/2dipole/ir/total_dipole_py.npy')
    print(np.mean(total_dipole, axis = 0))

    v_dipole = numerical_diff(total_dipole, dt)[500:-500]
    print(np.mean(v_dipole, axis = 0))
    v_dipole -= np.mean(v_dipole, axis = 0, keepdims = True)
    print(v_dipole[:5])
    
    corr = calculate_corr(v_dipole, v_dipole, 100000, 890000)
    print(corr[:10])
    print(corr[-10:])
    np.save('2deepwannier/2dipole/ir/corr_py.npy', corr)

    ir = calculate_ir(corr, width = 240, dt = dt, temperature = 300.)
    np.savetxt('2deepwannier/2dipole/ir/ft_py.dat', ir)

    