import copy
import os
import subprocess
import sys

import numpy as np
from scipy.fft import fft, ifft, fftshift
from scipy.io.wavfile import write, read
from scipy.signal import resample
from scipy.signal.windows import blackmanharris, triang

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), './utilFunctions_C/'))
try:
    import utilFunctions_C as UF_C
except ImportError:
    print("\n")
    print("-------------------------------------------------------------------------------")
    print("Warning:")
    print("Cython modules for some of the core functions were not imported.")
    print("for the instructions to compile the cython modules.")
    print("Exiting the code!!")
    print("-------------------------------------------------------------------------------")
    print("\n")
    sys.exit(0)

winsound_imported = False
if sys.platform == "win32":
    try:
        import winsound

        winsound_imported = True
    except:
        print("You won't be able to play sounds, winsound could not be imported")


def isPower2(num):
    return ((num & (num - 1)) == 0) and num > 0


INT16_FAC = (2 ** 15)
INT32_FAC = (2 ** 31)
INT64_FAC = (2 ** 63)
norm_fact = {'int16': INT16_FAC, 'int32': INT32_FAC, 'int64': INT64_FAC, 'float32': 1.0, 'float64': 1.0}


def wavread(filename):
    if (os.path.isfile(filename) == False):
        raise ValueError("Input file is wrong")

    fs, x = read(filename)

    if (len(x.shape) != 1):
        raise ValueError("Audio file should be mono")

    if ((fs != 44100) & (fs != 48000)):
        raise ValueError("Sampling rate of input sound should be 44100")

    x = np.float32(x) / norm_fact[x.dtype.name]
    return fs, x


def wavplay(filename):
    if (os.path.isfile(filename) == False):
        print("Input file does not exist. Make sure you computed the analysis/synthesis")
    else:
        if sys.platform == "linux" or sys.platform == "linux2":
            subprocess.call(["aplay", filename])
        elif sys.platform == "darwin":
            subprocess.call(["afplay", filename])
        elif sys.platform == "win32":
            if winsound_imported:
                winsound.PlaySound(filename, winsound.SND_FILENAME)
            else:
                print("Cannot play sound, winsound could not be imported")
        else:
            print("Platform not recognized")


def wavwrite(y, fs, filename):
    x = copy.deepcopy(y)
    x *= INT16_FAC
    x = np.int16(x)
    write(filename, fs, x)


def peakDetection(mX, t):
    thresh = np.where(np.greater(mX[1:-1], t), mX[1:-1], 0)
    next_minor = np.where(mX[1:-1] > mX[2:], mX[1:-1], 0)
    prev_minor = np.where(mX[1:-1] > mX[:-2], mX[1:-1], 0)
    ploc = thresh * next_minor * prev_minor
    ploc = ploc.nonzero()[0] + 1
    return ploc


def peakInterp(mX, pX, ploc):
    val = mX[ploc]
    lval = mX[ploc - 1]
    rval = mX[ploc + 1]
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)
    ipphase = np.interp(iploc, np.arange(0, pX.size), pX)
    return iploc, ipmag, ipphase


def sinc(x, N):
    y = np.sin(N * x / 2) / np.sin(x / 2)
    y[np.isnan(y)] = N
    return y


def genBhLobe(x):
    N = 512
    f = x * np.pi * 2 / N
    df = 2 * np.pi / N
    y = np.zeros(x.size)
    consts = [0.35875, 0.48829, 0.14128, 0.01168]
    for m in range(0, 4):
        y += consts[m] / 2 * (sinc(f - df * m, N) + sinc(f + df * m, N))
    y = y / N / consts[0]
    return y


def genSpecSines(ipfreq, ipmag, ipphase, N, fs):
    Y = UF_C.genSpecSines(N * ipfreq / float(fs), ipmag, ipphase, N)
    return Y


def genSpecSines_p(ipfreq, ipmag, ipphase, N, fs):
    Y = np.zeros(N, dtype=complex)
    hN = N // 2
    for i in range(0, ipfreq.size):
        loc = N * ipfreq[i] / fs
        if loc == 0 or loc > hN - 1:
            continue
        binremainder = round(loc) - loc
        lb = np.arange(binremainder - 4, binremainder + 5)
        lmag = genBhLobe(lb) * 10 ** (ipmag[i] / 20)
        b = np.arange(round(loc) - 4, round(loc) + 5, dtype='int')
        for m in range(0, 9):
            if b[m] < 0:
                Y[-b[m]] += lmag[m] * np.exp(-1j * ipphase[i])
            elif b[m] > hN:
                Y[b[m]] += lmag[m] * np.exp(-1j * ipphase[i])
            elif b[m] == 0 or b[m] == hN:
                Y[b[m]] += lmag[m] * np.exp(1j * ipphase[i]) + lmag[m] * np.exp(-1j * ipphase[i])
            else:
                Y[b[m]] += lmag[m] * np.exp(1j * ipphase[i])
        Y[hN + 1:] = Y[hN - 1:0:-1].conjugate()
    return Y


def sinewaveSynth(freqs, amp, H, fs):
    t = np.arange(H) / float(fs)
    lastphase = 0
    lastfreq = freqs[0]
    y = np.array([])
    for l in range(freqs.size):
        if (lastfreq == 0) & (freqs[l] == 0):
            A = np.zeros(H)
            freq = np.zeros(H)
        elif (lastfreq == 0) & (freqs[l] > 0):
            A = np.arange(0, amp, amp / H)
            freq = np.ones(H) * freqs[l]
        elif (lastfreq > 0) & (freqs[l] > 0):
            A = np.ones(H) * amp
            if (lastfreq == freqs[l]):
                freq = np.ones(H) * lastfreq
            else:
                freq = np.arange(lastfreq, freqs[l], (freqs[l] - lastfreq) / H)
        elif (lastfreq > 0) & (freqs[l] == 0):
            A = np.arange(amp, 0, -amp / H)
            freq = np.ones(H) * lastfreq
        phase = 2 * np.pi * freq * t + lastphase
        yh = A * np.cos(phase)
        lastfreq = freqs[l]
        lastphase = np.remainder(phase[H - 1], 2 * np.pi)
        y = np.append(y, yh)
    return y


def cleaningTrack(track, minTrackLength=3):
    nFrames = track.size
    cleanTrack = np.copy(track)
    trackBegs = np.nonzero((track[:nFrames - 1] <= 0) & (track[1:] > 0))[0] + 1
    if track[0] > 0:
        trackBegs = np.insert(trackBegs, 0, 0)
    trackEnds = np.nonzero((track[:nFrames - 1] > 0) & (track[1:] <= 0))[0] + 1
    if track[nFrames - 1] > 0:
        trackEnds = np.append(trackEnds, nFrames - 1)
    trackLengths = 1 + trackEnds - trackBegs
    for i, j in zip(trackBegs, trackLengths):
        if j <= minTrackLength:
            cleanTrack[i:i + j] = 0
    return cleanTrack


def f0Twm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0):
    if (minf0 < 0):
        raise ValueError("Minimum fundamental frequency (minf0) smaller than 0")

    if (maxf0 >= 10000):
        raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")

    if (pfreq.size < 3) & (f0t == 0):
        return 0

    f0c = np.argwhere((pfreq > minf0) & (pfreq < maxf0))[:, 0]
    if (f0c.size == 0):
        return 0
    f0cf = pfreq[f0c]
    f0cm = pmag[f0c]

    if f0t > 0:
        shortlist = np.argwhere(np.abs(f0cf - f0t) < f0t / 2.0)[:, 0]
        maxc = np.argmax(f0cm)
        maxcfd = f0cf[maxc] % f0t
        if maxcfd > f0t / 2:
            maxcfd = f0t - maxcfd
        if (maxc not in shortlist) and (maxcfd > (f0t / 4)):
            shortlist = np.append(maxc, shortlist)
        f0cf = f0cf[shortlist]

    if (f0cf.size == 0):
        return 0

    f0, f0error = UF_C.twm(pfreq, pmag, f0cf)
    if (f0 > 0) and (f0error < ef0max):
        return f0
    else:
        return 0


def TWM_p(pfreq, pmag, f0c):
    p = 0.5
    q = 1.4
    r = 0.5
    rho = 0.33
    Amax = max(pmag)
    maxnpeaks = 10
    harmonic = np.matrix(f0c)
    ErrorPM = np.zeros(harmonic.size)
    MaxNPM = min(maxnpeaks, pfreq.size)
    for i in range(0, MaxNPM):
        difmatrixPM = harmonic.T * np.ones(pfreq.size)
        difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1)) * pfreq)
        FreqDistance = np.amin(difmatrixPM, axis=1)
        peakloc = np.argmin(difmatrixPM, axis=1)
        Ponddif = np.array(FreqDistance) * (np.array(harmonic.T) ** (-p))
        PeakMag = pmag[peakloc]
        MagFactor = 10 ** ((PeakMag - Amax) / 20)
        ErrorPM = ErrorPM + (Ponddif + MagFactor * (q * Ponddif - r)).T
        harmonic = harmonic + f0c

    ErrorMP = np.zeros(harmonic.size)
    MaxNMP = min(maxnpeaks, pfreq.size)
    for i in range(0, f0c.size):
        nharm = np.round(pfreq[:MaxNMP] / f0c[i])
        nharm = (nharm >= 1) * nharm + (nharm < 1)
        FreqDistance = abs(pfreq[:MaxNMP] - nharm * f0c[i])
        Ponddif = FreqDistance * (pfreq[:MaxNMP] ** (-p))
        PeakMag = pmag[:MaxNMP]
        MagFactor = 10 ** ((PeakMag - Amax) / 20)
        ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor * (q * Ponddif - r)))

    Error = (ErrorPM[0] / MaxNPM) + (rho * ErrorMP / MaxNMP)
    f0index = np.argmin(Error)
    f0 = f0c[f0index]

    return f0, Error[f0index]


def sineSubtraction(x, N, H, sfreq, smag, sphase, fs):
    hN = N // 2
    x = np.append(np.zeros(hN), x)
    x = np.append(x, np.zeros(hN))
    bh = blackmanharris(N)
    w = bh / sum(bh)
    sw = np.zeros(N)
    sw[hN - H:hN + H] = triang(2 * H) / w[hN - H:hN + H]
    L = sfreq.shape[0]
    xr = np.zeros(x.size)
    pin = 0
    for l in range(L):
        xw = x[pin:pin + N] * w
        X = fft(fftshift(xw))
        Yh = UF_C.genSpecSines(N * sfreq[l, :] / fs, smag[l, :], sphase[l, :], N)
        Xr = X - Yh
        xrw = np.real(fftshift(ifft(Xr)))
        xr[pin:pin + N] += xrw * sw
        pin += H
    xr = np.delete(xr, range(hN))
    xr = np.delete(xr, range(xr.size - hN, xr.size))
    return xr


def stochasticResidualAnal(x, N, H, sfreq, smag, sphase, fs, stocf):
    hN = N // 2
    x = np.append(np.zeros(hN), x)
    x = np.append(x, np.zeros(hN))
    bh = blackmanharris(N)
    w = bh / sum(bh)
    L = sfreq.shape[0]
    pin = 0
    for l in range(L):
        xw = x[pin:pin + N] * w
        X = fft(fftshift(xw))
        Yh = UF_C.genSpecSines(N * sfreq[l, :] / fs, smag[l, :], sphase[l, :], N)
        Xr = X - Yh
        mXr = 20 * np.log10(abs(Xr[:hN]))
        mXrenv = resample(np.maximum(-200, mXr), mXr.size * stocf)
        if l == 0:
            stocEnv = np.array([mXrenv])
        else:
            stocEnv = np.vstack((stocEnv, np.array([mXrenv])))
        pin += H
    return stocEnv
