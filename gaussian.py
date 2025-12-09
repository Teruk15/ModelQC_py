from scipy.io import savemat
import numpy as np
import os
import sys

dataPath = './datasets/npz/data_feature.npz'
savePath = './model'

# 1. Load data_feature.npz
if not os.path.exists(dataPath):
    print(f'{dataPath} does not exist')
    sys.exit(1)

data = np.load(dataPath)

# We only need features (no labels)
X_train = data['X_train']   # clean / good samples
X_test  = data['X_test']    # held-out samples (also assumed good, used for sanity check)

print("Total train samples:", X_train.shape[0])

if X_train.shape[0] < 10:
    print("Too few samples to fit Gaussian model.")
    sys.exit(1)

# 2. Estimate mean and covariance of clean features
mu = X_train.mean(axis=0)             # shape (n_feat,)
Sigma = np.cov(X_train, rowvar=False) # shape (n_feat, n_feat)

# (Optional) regularize covariance if near-singular
eps = 1e-6
Sigma_reg = Sigma + eps * np.eye(Sigma.shape[0])

# Precompute inverse for Mahalanobis distance
Sigma_inv = np.linalg.inv(Sigma_reg)

# 3. Compute squared Mahalanobis distance for train samples
diff_train = X_train - mu
d2_train = np.sum(diff_train @ Sigma_inv * diff_train, axis=1)

# 4. Choose threshold as a percentile of train distances
percentile = 97.5
threshold = np.percentile(d2_train, percentile)

print(f"Chosen threshold (sq. Mahalanobis) at {percentile}th percentile: {threshold:.4f}")

# 5. (Optional) sanity check on X_test
diff_test = X_test - mu
d2_test = np.sum(diff_test @ Sigma_inv * diff_test, axis=1)
y_pred_test = (d2_test > threshold).astype(int)  # 1 = flagged as bad

print("Test windows flagged as bad:", int(y_pred_test.sum()))
print("Test windows total:", X_test.shape[0])
print("Fraction flagged (test):", float(y_pred_test.mean()))

# 6. Save parameters for later use
if not os.path.exists(savePath):
    print(f'{savePath} does not exist')
    sys.exit(1)

saveFilePath = os.path.join(savePath, 'gaussian_model.npz')
np.savez_compressed(
    saveFilePath,
    mu=mu,
    Sigma=Sigma_reg,
    Sigma_inv=Sigma_inv,
    threshold=threshold
)

print(f"Saved Gaussian model parameters to {saveFilePath}")

# Also save as .mat for MATLAB / Simulink
savemat(
    os.path.join(savePath, 'gaussian_model.mat'),
    {
        'mu': mu,
        'Sigma_inv': Sigma_inv,
        'threshold': float(threshold),
    }
)

print("Saved MATLAB version to ./model/gaussian_model.mat")