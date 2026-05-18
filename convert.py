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
import numpy as np

# Expected .mat structure
# DATA.X = [channel x sample]
# DATA.y = [channel x 1]

def main():
    datasetPath = "./datasets/mat"
    savePath = "./datasets/npz"
    
    test_file = "data001.mat"

    Xs = []  # All x
    ys = []  # All y
    
    X_test = None # Reserved patient data, X
    y_test = None # Reserved patient label, y
    
    window_sizes = []  # N_window tracker

    window_length = 4800
    ds_factor = 6

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
        print(f'Loading {fullpath}...')
        varname = "DATA"

        X: np.ndarray = data[varname]["X"][0, 0]
        y: np.ndarray = data[varname]["y"][0, 0]  # Accessing y (label)
        y = y.reshape(-1)  # 2D -> 1D

        # Conversion: X -> [N_window, N_window_sample], y -> [N_window,1]
        X, y = windowResize(X, y, window_length, ds_factor)
        
        # If this is test set save in different memory
        if file == test_file:
            X_test = X
            y_test = y
            continue

        # Track each patient window sample count to handle patient imbalance later
        window_sizes.append(y.shape[0])

        Xs.append(X)
        ys.append(y)

    window_limit = min(window_sizes)  # This can be a constant as well (e.g. 1000)

    # Handle patient-imbalance (cap to minimum window size)
    for i, window_size in enumerate(window_sizes):

        if window_size <= window_limit:
            continue  # or keep all windows
        
        # Random index pick
        idxs = np.random.choice(window_size, size=window_limit, replace=False)

        Xs[i] = Xs[i][idxs, :]
        ys[i] = ys[i][idxs]
    
    X_all = np.vstack(Xs) 
    y_all = np.concatenate(ys)
    
    print(X_all.shape, y_all.shape)
    
    mask_true  = (y_all == True)
    mask_false = (y_all == False)
    mask_other = ~(mask_true | mask_false)

    print("True:", mask_true.sum(), "False:", mask_false.sum(), "Other:", mask_other.sum())
    print("Other unique values:", np.unique(y_all[mask_other]))
    print("Other indices:", np.where(mask_other)[0][:20])
    
    # Saving as .npz file
    if not os.path.exists(savePath):
        print(f'{savePath} does not exist')
        sys.exit(1)
        
    saveFilePath = os.path.join(savePath, 'data')
    np.savez_compressed(saveFilePath, X=X_all, y=y_all, X_test=X_test, y_test=y_test)
    
    print(f'Number of patient (training): {len(window_sizes)},\n\
            X-size: {X_all.shape},\n\
            y-size: {y_all.shape},\n\
            X-test: {X_test.shape},\n\
            y-test: {y_test.shape}')
    
    print(f'Saved as {saveFilePath}.npz')


# def windowResize(X: np.ndarray, y: np.ndarray, window_length, ds_factor):
#     # C = N_channel
#     # N = N_sample
#     # W = N_window_per_channel

#     C, N = X.shape
#     W = math.floor(N / window_length)

#     X = X[:, 0 : W * window_length]  # Permute only required X

#     Xw = X.reshape(C, W, window_length)

#     X_resized = Xw.reshape(-1, Xw.shape[2])  # 2D: [N_window, N_window_sample]

#     y_resized = y.repeat(W)  # 1D: [N_window, 1]

#     return X_resized, y_resized

def windowResize(X: np.ndarray, y: np.ndarray, window_length, ds_factor):
    """
    window_length: samples per 1 second at ORIGINAL fs (4800)
    ds_factor: downsample factor
    """

    # Downsample in time
    if ds_factor > 1:
        X = X[:, ::ds_factor]

    # After downsampling, 1-second window has fewer samples
    # (requires window_length divisible by ds_factor)
    effective_window_length = window_length // ds_factor

    C, N = X.shape
    W = N // effective_window_length

    X = X[:, 0 : W * effective_window_length]
    Xw = X.reshape(C, W, effective_window_length)

    X_resized = Xw.reshape(-1, effective_window_length)  # [N_windows_total, samples_per_1sec]
    y_resized = y.repeat(W)

    return X_resized, y_resized
    

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
