"""
Quick Description (revised later):
1. Convert from .mat -> .npz (possibly with down-sample)
2. Save as .npz
3. Optionally save as csv to quick view of the data
"""

from scipy.io import loadmat
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Expected .mat structure
# DATA.X = [channel x sample]
# DATA.y = [channel x 1]

def main():
    datasetPath = "./datasets/mat"
    savePath = "./datasets/npz"

    Xs = []  # All x
    ys = []  # All y
    windowSizes = []  # N_window tracker

    windowLength = 4800

    # Directory check
    if not os.path.exists(datasetPath):
        print(f"{datasetPath} does not exist")
        sys.exit(1)

    # Extract .mat files
    for file in os.listdir(datasetPath):
        if not file.lower().endswith(".mat"):
            continue

        fullpath = os.path.join(datasetPath, file)
        data = loadmat(fullpath)
        varname = "DATA"

        X: np.ndarray = data[varname]["X"][0, 0]
        y: np.ndarray = data[varname]["y"][0, 0]  # Accessing y (label)
        y = y.reshape(-1)  # 2D -> 1D

        # Convert to [N_window, N_window_sample]
        X, y = windowResize(X, y, windowLength)

        # Track each patient window sample count to handle patient imbalance later
        windowSizes.append(y.shape[0])

        Xs.append(X)
        ys.append(y)

    windowLimit = min(windowSizes)  # This can be a constant as well (e.g. 1000)

    # Handle patient-imbalance (cap to minimum window size)
    for i, windowSize in enumerate(windowSizes):

        if windowSize <= windowLimit:
            continue  # or keep all windows
        
        # Random index pick
        idxs = np.random.choice(windowSize, size=windowLimit, replace=False)

        Xs[i] = Xs[i][idxs, :]
        ys[i] = ys[i][idxs]
    
    X_all = np.vstack(Xs) 
    y_all = np.concatenate(ys)
    
    # Running FFT analysis (True == noisy, False = clean)
    # fs = 4800
    # for i in range(5):
    #     frequencyAnalysis(X_all, y_all, False, fs)
    #     plt.close()
    
    # Saving as .npz file
    if not os.path.exists(savePath):
        print(f'{savePath} does not exist')
        sys.exit(1)
        
    saveFilePath = os.path.join(savePath, 'data')
    np.savez_compressed(saveFilePath, X=X_all, y=y_all)
    
    print(f'Saved as {saveFilePath}.npz')


def windowResize(X: np.ndarray, y: np.ndarray, windowLength):
    # C = N_channel
    # N = N_sample
    # W = N_window_per_channel

    C, N = X.shape
    W = math.floor(N / windowLength)

    X = X[:, 0 : W * windowLength]  # Permute only required X

    Xw = X.reshape(C, W, windowLength)

    X_resized = Xw.reshape(-1, Xw.shape[2])  # 2D: [N_window, N_window_sample]

    y_resized = y.repeat(W)  # 1D: [N_window, 1]

    return X_resized, y_resized

def frequencyAnalysis(X_all: np.ndarray, y_all: np.ndarray, label: bool, fs):
    idxAll = np.where(y_all == label)[0]
    randIdx = np.random.choice(idxAll, size=1, replace=False)
    
    f_max = 500 # Change as needed
    
    X = X_all[randIdx, :]
    
    N = X.shape[1]
    
    Xf = np.fft.rfft(X, axis=1)  # Compute fft
    
    Xmag = np.abs(Xf) / N
    freqs = np.fft.rfftfreq(N, d=1/fs)
    XmagMean = Xmag.mean(axis=0)
    
    mask = freqs <= f_max
    
    freqs_cut = freqs[mask]
    XmagMean_cut = XmagMean[mask]
    
    plt.figure()
    plt.plot(freqs_cut, XmagMean_cut)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'FFT Magnitude Spectrum for label == {"noisy" if label else "clean"}')
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    main()
    
    

# X_all = np.vstack(Xs)          # [total_windows, N_windowSample]
# print(f'Total X size: {X_all.shape}')

# # y_all = np.concatenate(ys)     # [total_windows]
# # print(f'Total y size: {y_all.shape}')

# if not os.path.exists(savePath):
#     print(f'{savePath} does not exist')
#     sys.exit(1)


# saveFilePath = os.path.join(savePath, 'data')
#  # Save as .npz file


# Visual check (optional)
# if visual:
#     data = np.load("./datasets/npz/data.npz", allow_pickle=True)
#     X = data["X"]
#     # y = data['y']

#     outPath = "./datasets/npz/optional.csv"

#     df = pd.concat([pd.DataFrame(X)], axis=1)
#     # df = pd.concat([pd.Series(y, name='label'), pd.DataFrame(X)], axis=1)

#     df.to_csv(outPath, index=False)
