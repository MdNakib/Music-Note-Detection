import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.signal import get_window
# from playsound import playsound
sys.path.append('../software/models/')
import utilFunctions as UF
import stft as STFT
import sine_track as ST
import sound_regenerated as SR
import verify as VF

# inputFile = '../sounds/C Maj C4.wav'
def main(inputFile):
    M = 11625
    N = 2 ** (int)(np.log2(M) + 1)
    H = M // 8

    (fs, x) = UF.wavread(inputFile)

    w = get_window('hamming', M)

    mX, pX = STFT.stftAnal(x, w, N, H)

    # playsound(inputFile)

    (numFrames, sizeOfFrame) = mX.shape
    freqs = np.zeros(numFrames)

    for i in range(numFrames):
        freqs[i] = np.argmax(mX[i, :]) * fs / N

    maxInFrame = np.zeros(numFrames)
    for k in range(numFrames):
        maxInFrame[k] = max(mX[k, :])


    def frequency_to_note(frequencies):
        # Define the list of notes and their corresponding frequencies
        note_freq_mapping = {
            'C2': 65.4, 'C#2': 69.3, 'D2': 73.4, 'D#2': 77.8,
            'E2': 82.4, 'F2': 87.3, 'F#2': 92.5, 'G2': 98,
            'G#2': 103.8, 'A2': 110, 'A#2': 116.5, 'B2': 123.5,
            'C3': 130.8, 'C#3': 138.6, 'D3': 146.8, 'D#3': 155.6,
            'E3': 164.8, 'F3': 174.6, 'F#3': 185, 'G3': 196,
            'G#3': 207.7, 'A3': 220, 'A#3': 233.1, 'B3': 246.9,
            'C4': 261.6, 'C#4': 277.2, 'D4': 293.7, 'D#4': 311.1,
            'E4': 329.6, 'F4': 349.2, 'F#4': 370, 'G4': 392,
            'G#4': 415.3, 'A4': 440, 'A#4': 466.2, 'B4': 493.9,
            'C5': 523.3, 'C#5': 554.4, 'D5': 587.3, 'D#5': 622.3,
            'E5': 659.3, 'F5': 698.5, 'F#5': 740, 'G5': 784,
            'G#5': 830.6, 'A5': 880, 'A#5': 932.3, 'B5': 987.8,
            'C5': 1046.5
        }

    # Find the note with the closest frequency for each frequency in the array
        closest_notes = [min(note_freq_mapping, key=lambda note: abs(note_freq_mapping[note] - freq)) for freq in
                     frequencies]

        return closest_notes


    plt.figure()
    plt.subplot(211)
    frmTime = H * np.arange(numFrames) / fs
    binFreqs = np.arange(N / 2 + 1) * fs / N
    plt.pcolormesh(frmTime, binFreqs, np.transpose(mX), cmap=plt.get_cmap('jet'))
    plt.ylim([0, 4000])
    plt.title('Spectrogram')
    plt.ylabel('frequency')
    plt.xlabel('time')

    plt.subplot(212)
    plt.plot(frmTime, maxInFrame)
    plt.ylabel('magnitude')
    plt.xlabel('time')

    thresholdUp = -30
    thresholdDown = -35
    Notes = np.empty(0)
    time_span = np.empty(0)

    restart = True

    for i in range(1, numFrames - 1):
        if ((maxInFrame[i] >= thresholdUp) & (maxInFrame[i] >= maxInFrame[i - 1]) & (
                maxInFrame[i] >= maxInFrame[i + 1]) & restart):
            Notes = np.append(Notes, freqs[i])
            restart = False
            # print(i)
            time_span = np.append(time_span, i * H / fs)
        if (maxInFrame[i] < thresholdDown):
            restart = True
    notes = frequency_to_note(Notes)
    # plt.ylim(0,4000)
    print(Notes)
    print(notes)
    print(time_span)

    SR.main(inputFile)  # running sound regeneration code

    plt.figure()
    ST.main(inputFile)  # running for sine tracking

    # for verification
    # outputFile = os.path.basename(inputFile)[:-4] + '_sineModel.wav'
    # signal1 = x
    # (fs, signal2) = UF.wavread(outputFile)
    # mse = VF.main(signal1, signal2)
    #
    # plt.figure()
    # sizes = [100 - mse, mse]
    # plt.pie(sizes)

plt.show()
