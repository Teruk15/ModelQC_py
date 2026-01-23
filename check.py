from numpy.linalg import inv
from scipy.stats import chi2, probplot
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

dataPath = "./datasets/npz/data_feature.npz"
savePath = "./model"

# Load data_feature.npz
if not os.path.exists(dataPath):
    print(f"{dataPath} does not exist")
    sys.exit(1)

data = np.load(dataPath)

# We only need features (no labels)
X_train = data["X_train"]  # clean / good samples

print(f"X_train shape: {X_train.shape}")

n_features = X_train.shape[1]

plot_feature_hist = False
compare_d2_x2 = False
check_covariance = True

# Check-1: Plot each feature distribution
if plot_feature_hist:
    for j in range(n_features):
        feature = X_train[:, j]

        plt.figure()
        plt.hist(feature, bins=50, density=True)
        plt.title(f"Feature {j}")
        plt.xlabel("value")
        plt.ylabel("density")
        plt.show()

# Check-2: Compare whether D^2 ~ X^2
if compare_d2_x2:
    mu = X_train.mean(axis=0)
    Sigma = np.cov(X_train, rowvar=False)
    Sigma_inv = inv(Sigma)

    # Mahalanobis distance squared
    D2 = np.sum((X_train - mu) @ Sigma_inv * (X_train - mu), axis=1)
    print("mean(D2) =", D2.mean())
    
    # Log likelihood
    ll = -0.5 * (D2 + np.log(np.linalg.det(Sigma)) + n_features*np.log(2*np.pi))
    print("mean(log-likelihood) =", ll.mean())

    plt.figure()
    plt.hist(D2, bins=50, density=True, alpha=0.6, label="Empirical")

    x = np.linspace(0, np.max(D2), 500)
    plt.plot(x, chi2.pdf(x, df=n_features), "r--", label=f"χ²(df={n_features})")

    plt.xlabel("Mahalanobis distance²")
    plt.ylabel("density")
    plt.legend()
    plt.title("Mahalanobis distance check")
    plt.show()

    plt.figure()
    probplot(D2, dist=chi2(df=n_features), plot=plt)
    plt.title("QQ plot: Mahalanobis² vs χ²")
    plt.show()

# Check-3: Covariance check
if check_covariance:
    corr = np.corrcoef(X_train, rowvar=False)

    plt.figure(figsize=(6, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="correlation")
    plt.xticks(range(n_features), range(n_features))
    plt.yticks(range(n_features), range(n_features))
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.show()
