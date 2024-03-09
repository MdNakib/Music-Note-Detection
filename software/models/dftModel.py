import numpy as np
import math
from scipy.fft import fft, ifft
import utilFunctions as UF

tol = 1e-14

def dftModel(x, w, N):
    if not (UF.isPower2(N)):
        raise ValueError("FFT size (N) is not a power of 2")

    if (w.size > N):
        raise ValueError("Window size (M) is bigger than FFT size")

    if all(x == 0):
        return np.zeros(x.size)
    hN = (N // 2) + 1
    hM1 = (w.size + 1) // 2
    hM2 = int(math.floor(w.size / 2))
    fftbuffer = np.zeros(N)
    y = np.zeros(x.size)
    xw = x * w
    fftbuffer[:hM1] = xw[hM2:]
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)
    absX = abs(X[:hN])
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps
    mX = 20 * np.log10(absX)
    pX = np.unwrap(np.angle(X[:hN]))
    Y = np.zeros(N, dtype=complex)
    Y[:hN] = 10 ** (mX / 20) * np.exp(1j * pX)
    Y[hN:] = 10 ** (mX[-2:0:-1] / 20) * np.exp(-1j * pX[-2:0:-1])
    fftbuffer = np.real(ifft(Y))
    y[:hM2] = fftbuffer[-hM2:]
    y[hM2:] = fftbuffer[:hM1]
    return y

def dftAnal(x, w, N):
    if not (UF.isPower2(N)):
        raise ValueError("FFT size (N) is not a power of 2")

    if w.size > N:
        raise ValueError("Window size (M) is bigger than FFT size")

    hN = (N // 2) + 1
    hM1 = (w.size + 1) // 2
    hM2 = w.size // 2
    fftbuffer = np.zeros(N)
    w = w / sum(w)
    xw = x * w
    fftbuffer[:hM1] = xw[hM2:]
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)
    absX = abs(X[:hN])
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps
    mX = 20 * np.log10(absX)
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0
    pX = np.unwrap(np.angle(X[:hN]))
    return mX, pX

def dftSynth(mX, pX, M):
    hN = mX.size
    N = (hN - 1) * 2
    if not (UF.isPower2(N)):
        raise ValueError("size of mX is not (N/2)+1")

    hM1 = int(math.floor((M + 1) / 2))
    hM2 = int(math.floor(M / 2))
    y = np.zeros(M)
    Y = np.zeros(N, dtype=complex)
    Y[:hN] = 10 ** (mX / 20) * np.exp(1j * pX)
    Y[hN:] = 10 ** (mX[-2:0:-1] / 20) * np.exp(-1j * pX[-2:0:-1])
    fftbuffer = np.real(ifft(Y))
    y[:hM2] = fftbuffer[-hM2:]
    y[hM2:] = fftbuffer[:hM1]
    return y
