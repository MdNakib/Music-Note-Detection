import numpy as np
from scipy.signal import correlate

def MSE(signal1, signal2):
    # Calculate Mean Squared Error (MSE)

    len1 = len(signal1)
    len2 = len(signal2)

    if len1 > len2:
        excess_length = (len1 - len2) // 2
        signal1 = signal1[excess_length:len1 - excess_length]
    elif len2 > len1:
        excess_length = (len2 - len1) // 2
        signal2 = signal2[excess_length:len2 - excess_length]

    if len(signal1) < len(signal2):
        signal1 = np.append(signal1,0)
    elif len(signal2) < len(signal1):
        signal2 = np.append(signal2,0)
    return  np.mean((signal1 - signal2)**2) * 100

def cosine_similarity(signal1,signal2):
    N = len(signal1)
    M = len(signal2)
    print(N)
    print(M)
    L = M - N
    if (L >= 0):
        if (np.mod(L, 2) == 0):
            signal2 = signal2[L // 2:-L // 2]
        else:
            signal2 = signal2[L // 2:-L // 2 + 1]
    else:
        L = abs(L)
        if (np.mod(L, 2) == 0):
            signal1 = signal1[L // 2:-L // 2]
        else:
            signal1 = signal1[L // 2:-L // 2 + 1]

    dot_product = np.dot(signal1, signal2)
    norm_vector1 = np.linalg.norm(signal1)
    norm_vector2 = np.linalg.norm(signal2)
    similarity = dot_product / (norm_vector1 * norm_vector2) * 100
    return similarity

