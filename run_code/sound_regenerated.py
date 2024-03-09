import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../software/models/'))
import utilFunctions as UF
import sineModel as SM

def main(inputFile = '../sounds/C Maj C4.wav', window = 'hamming', M = 2001, N = 2048, t = -80, minSineDur = 0.02, maxnSines = 150, freqDevOffset = 10, freqDevSlope = 0.001):
    Ns = 512
    H = 128

    (fs,x) = UF.wavread(inputFile)
    print(Ns)

    w = get_window(window, M)

    tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

    y = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

    outputFile = os.path.basename(inputFile)[:-4] + '_sineModel.wav'

    UF.wavwrite(y,fs,outputFile)
    print(tfreq.size)

    print('Successful')
