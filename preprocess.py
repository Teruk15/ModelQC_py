from scipy.signal import butter, filtfilt
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    dataPath = './datasets/npz/data.npz'
    savePath = './datasets/npz'

    # Load data
    if not os.path.exists(dataPath):
        print(f'{dataPath} does not exist')
        sys.exit(1)

    data = np.load(dataPath)
    
    X: np.ndarray = data["X"] # [N_window, N_window_sample]
    X_test: np.ndarray = data["X_test"]
    
    y: np.ndarray = data["y"]
    y_test: np.ndarray = data["y_test"]
    
    # fs = 4800
    # clean_mask = (y == False)
    # X[clean_mask, :] = add_line_noise_2d(X[clean_mask, :], fs, freq=60, alpha=1)
    # clean_mask = (y_test == False)
    # X_test[clean_mask, :] = add_line_noise_2d(X_test[clean_mask, :], fs, freq=60, alpha=1)

    # Normalization (DC-offset + z-score)
    X = X - X.mean(axis=1, keepdims=True)
    X = X / (X.std(axis=1, keepdims=True) + 1e-8)
    X_test = X_test - X_test.mean(axis=1, keepdims=True)
    X_test = X_test / (X_test.std(axis=1, keepdims=True) + 1e-8)
    
    # Running FFT analysis for debugging (True == noisy, False = clean)
    fs = 4800
    frequencyAnalysis(X, y, False, fs)
    frequencyAnalysis(X, y, True, fs)
    
    Bands = [
        [1, 10],     # slow drift / instability (but excludes DC)
        [55, 65],    # line noise around 60 Hz (very discriminative)
        [150, 250],  # high-frequency junk / broadband noise floor
    ]
    fs = 4800
    F = len(Bands)
    
    # Assumption of X's shape: [L, W]
    #   L = number of windows    
    #   W = sample of window
    #   F = number of frequency bands
    
    # 1). Preprocessing for Training sets
    L, W = X.shape
    X_out = np.zeros((F, W, L))
    
    for l in range(L):
        # Bandpass rows (log-normalization)
        for f, (f_lo, f_hi) in enumerate(Bands):
            # band = bandpass_filter(X[l, :], fs, f_lo, f_hi)
            # band = np.log(band**2 + 1e-8)
            # band = (band - band.mean()) / (band.std() + 1e-8)
            X_out[f, :, l] = bandpass_filter(X[l, :], fs, f_lo, f_hi)
    
    
    # 2). Preprocessing for Testing sets
    L, W = X_test.shape
    X_out_test = np.zeros((F, W, L))
    
    for l in range(L):
        # Bandpass rows (log-normalization)
        for f, (f_lo, f_hi) in enumerate(Bands):
            # band = bandpass_filter(X_test[l, :], fs, f_lo, f_hi)
            # band = np.log(band**2 + 1e-8)
            # band = (band - band.mean()) / (band.std() + 1e-8)
            X_out_test[f, :, l] = bandpass_filter(X_test[l, :], fs, f_lo, f_hi)
    
    
    # Sanity check: filtered data must be finite and non-trivial
    assert np.isfinite(X_out).all(), "Filtered training data contains NaN or Inf"
    assert np.isfinite(X_out_test).all(), "Filtered testing data contains NaN or Inf"
    assert np.mean(np.std(X_out, axis=1)) > 1e-6, "Filtered training data is near-zero everywhere"
    assert np.mean(np.std(X_out_test, axis=1)) > 1e-6, "Filtered testing data is near-zero everywhere"
    
    # Size of X_out   = one window x len(windows)
    # [F,W,L]         = [F,W]      x [L] 
    # Saving as .npz file
    if not os.path.exists(savePath):
        print(f'{savePath} does not exist')
        sys.exit(1)
        
    saveFilePath = os.path.join(savePath, 'data_preprocessed')
    np.savez_compressed(saveFilePath, X=X_out, y=y, X_test=X_out_test, y_test=y_test)
    
    print(f'Result of preprocessing (Bands = {Bands}): ,\n\
            X-size: {X_out.shape},\n\
            y-size: {y.shape},\n\
            X-test: {X_out_test.shape},\n\
            y-test: {y_test.shape}')
    
    print(f'Saved as {saveFilePath}.npz')
 
def bandpass_filter(x, fs, f_lo, f_hi, order=3):
    nyq = fs / 2
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype='band')
    return filtfilt(b, a, x)

def frequencyAnalysis(X_all: np.ndarray, y_all: np.ndarray, label: bool, fs):
    idx_all = np.where(y_all == label)[0]
    print("label", label, "count", len(idx_all))

    f_max = 500  # Hz

    # Take ALL windows with this label
    X = X_all[idx_all, :]  # [num_windows, N]

    N = X.shape[1]

    # FFT per window
    Xf = np.fft.rfft(X, axis=1)

    # Magnitude spectrum
    X_mag = np.abs(Xf) / N  # [num_windows, N_freq]

    # Mean & std across windows
    X_mag_mean = X_mag.mean(axis=0)
    X_mag_std  = X_mag.std(axis=0)

    freqs = np.fft.rfftfreq(N, d=1/fs)
    mask = freqs <= f_max

    name = "noisy" if label else "clean"

    # Plot 1: mean only (unchanged)
    plt.figure()
    plt.plot(freqs[mask], X_mag_mean[mask])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Average FFT ({name})')
    plt.grid(True)
    plt.show()
    
    plt.close

    # Plot 2: mean ± std
    plt.figure()
    plt.plot(freqs[mask], X_mag_mean[mask], label="mean")
    plt.fill_between(
        freqs[mask],
        (X_mag_mean - X_mag_std)[mask],
        (X_mag_mean + X_mag_std)[mask],
        alpha=0.3,
        label="±1 std"
    )
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Average FFT ± std ({name})')
    plt.grid(True)
    plt.legend()
    plt.show()

def add_line_noise_2d(X2d, fs, freq=60, alpha=0.05):
    """
    X2d: [num_windows, W]
    Adds a 60 Hz sinusoid to each window, scaled by that window's std.
    """
    W = X2d.shape[1]
    t = np.arange(W) / fs                      # [W]
    s = np.sin(2 * np.pi * freq * t)[None, :]  # [1, W]

    # per-window scaling
    scale = alpha * X2d.std(axis=1, keepdims=True)  # [num_windows, 1]
    return X2d + scale * s



if __name__ == "__main__":
    main()