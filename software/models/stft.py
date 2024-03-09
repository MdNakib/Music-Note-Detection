import numpy as np
import dftModel as DFT

def stft(x, w, N, H):
    if (H <= 0):
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size
    hM1 = (M + 1) // 2
    hM2 = M // 2
    x = np.append(np.zeros(hM2), x)
    x = np.append(x, np.zeros(hM1))
    pin = hM1
    pend = x.size - hM1
    w = w / sum(w)
    y = np.zeros(x.size)
    while pin <= pend:
        x1 = x[pin - hM1:pin + hM2]
        mX, pX = DFT.dftAnal(x1, w, N)
        y1 = DFT.dftSynth(mX, pX, M)
        y[pin - hM1:pin + hM2] += H * y1
        pin += H
    y = np.delete(y, range(hM2))
    y = np.delete(y, range(y.size - hM1, y.size))
    return y

def stftAnal(x, w, N, H):
    if (H <= 0):
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size
    hM1 = (M + 1) // 2
    hM2 = M // 2
    x = np.append(np.zeros(hM2), x)
    x = np.append(x, np.zeros(hM2))
    pin = hM1
    pend = x.size - hM1
    w = w / sum(w)
    xmX = []
    xpX = []
    while pin <= pend:
        x1 = x[pin - hM1:pin + hM2]
        mX, pX = DFT.dftAnal(x1, w, N)
        xmX.append(np.array(mX))
        xpX.append(np.array(pX))
        pin += H
    xmX = np.array(xmX)
    xpX = np.array(xpX)
    return xmX, xpX

def stftSynth(mY, pY, M, H):
    hM1 = (M + 1) // 2
    hM2 = M // 2
    nFrames = mY[:, 0].size
    y = np.zeros(nFrames * H + hM1 + hM2)
    pin = hM1
    for i in range(nFrames):
        y1 = DFT.dftSynth(mY[i, :], pY[i, :], M)
        y[pin - hM1:pin + hM2] += H * y1
        pin += H
    y = np.delete(y, range(hM2))
    y = np.delete(y, range(y.size - hM1, y.size))
    return y
