import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os,sys
sys.path.append('../software/models/')
import utilFunctions as UF
import sineModel as SM

# inputFile = '../sounds/C Maj C4.wav'
def main(inputFile = '../sounds/C Maj C4.wav'):
    window = 'hamming'
    M = 501
    N = 2 ** 10
    t = -90
    minSineDur = 0.02
    maxnSines = 20
    freqDevOffset = 10
    freqDevSlope = 0.001
    H = 200

    (fs,x) = UF.wavread(inputFile)
    w = get_window(window, M)
    tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

    numFrames = int(tfreq[:,0].size)
    # print(tmag)
    frmTime = H * np.arange(numFrames)/ fs;
    tfreq[tfreq<=0] = np.nan
    # plt.plot(frmTime, tfreq)
    plt.plot(frmTime, tfreq)
    plt.ylim(0,4000)
    # print(tfreq[tfreq >0])
    plt.show()