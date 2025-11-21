from scipy.stats import kurtosis
import os
import sys
import numpy as np

# X_ = [n_window x n_sample]
# y_ = [n_window x 1]

dataPath = './datasets/npz/data_split.npz'
savePath = './datasets/npz'

# Load data
if not os.path.exists(dataPath):
    print(f'{dataPath} does not exist')
    sys.exit(1)

data = np.load(dataPath)

X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']

y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

# Transform X_ into features
def transform(X):
    features = [extract_features(win) for win in X]
    return np.vstack(features)

# Helper for transform function
# THIS IS MOST IMPORTANT IN THE MODEL
def extract_features(window):
     # 1. Mean
    mean_val = np.mean(window)

    # 2. Variance
    var_val = np.var(window)

    # 3. Linear trend slope (simple least squares)
    t = np.arange(len(window))
    slope = np.polyfit(t, window, 1)[0]   # only slope coefficient

    # 4. Peak-to-peak amplitude
    p2p = np.ptp(window)   # max(window) - min(window)

    # 5. Kurtosis
    kurt = kurtosis(window, fisher=False)   # fisher=False gives normal=3.0

    # 6. Max derivative (detect pops)
    dx = np.abs(np.diff(window))
    max_deriv = np.max(dx)

    return np.array([
        mean_val,
        var_val,
        slope,
        p2p,
        kurt,
        max_deriv
    ])

X_train_feat = transform(X_train)
X_val_feat   = transform(X_val)
X_test_feat  = transform(X_test)

# Save
if not os.path.exists(savePath):
    print(f'{savePath} does not exist')
    sys.exit(1)

saveFilePath = os.path.join(savePath, 'data_feature')
np.savez_compressed(saveFilePath,
                    X_train=X_train_feat,
                    X_val=X_val_feat,
                    X_test=X_test_feat,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test)