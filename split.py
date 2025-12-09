'''
Quick Description (revised later):
1. Impute the X by row-mean
2. Split X & y into train, validation, test set
3. Save into single .npz file
'''

from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

dataPath = './datasets/npz/data.npz'
savePath = './datasets/npz'

# Load data
if not os.path.exists(dataPath):
    print(f'{dataPath} does not exist')
    sys.exit(1)

data = np.load(dataPath)
X = data["X"]    # [total_windows, N_sample]

# Train vs Test (0.8-to-0.2)
X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42, shuffle=True
)

print("Train:", X_train.shape)
print("Test: ", X_test.shape)

# Save
if not os.path.exists(savePath):
    print(f'{savePath} does not exist')
    sys.exit(1)

saveFilePath = os.path.join(savePath, 'data_split')

np.savez_compressed(
    saveFilePath,
    X_train=X_train,
    X_test=X_test
)



# # Load data
# if not os.path.exists(dataPath):
#     print(f'{dataPath} does not exist')
#     sys.exit(1)
    
# data:np.lib.npyio.NpzFile = np.load(dataPath, allow_pickle=True)
# X = data["X"]
# y = data["y"]

# # Impute by rows (no leakage)
# row_mean = np.nanmean(X, axis=1)
# inds = np.where(np.isnan(X))
# X[inds] = np.take(row_mean, inds[0])
# print(f'Missing data indices size: {inds}')

# # 1). Split X into 80% train, 20% test
# X_train_val, X_test, y_train_val, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, shuffle=True
# )

# # 2). Split X_train into 87.5% train, 12.5% test
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_val, y_train_val, test_size=0.125, random_state=42, shuffle=True
# )

# # Show sizes
# print("Total shape:", X.shape)
# print(f"Train shape ({round(X_train.shape[0] / X.shape[0] * 100, 1)}%):", X_train.shape)
# print(f"Validation shape ({round(X_val.shape[0] / X.shape[0] * 100, 1)}%):", X_val.shape)
# print(f"Test shape ({round(X_test.shape[0] / X.shape[0] * 100, 1)}%):", X_test.shape)


# # Save
# if not os.path.exists(savePath):
#     print(f'{savePath} does not exist')
#     sys.exit(1)

# saveFilePath = os.path.join(savePath, 'data_split')
# np.savez_compressed(saveFilePath,
#                     X_train=X_train,
#                     X_val=X_val,
#                     X_test=X_test,
#                     y_train=y_train,
#                     y_val=y_val,
#                     y_test=y_test)
