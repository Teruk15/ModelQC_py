'''
Quick Description (revised later):
1. Convert from .mat -> .npz (possibly with down-sample)
2. Save as .npz
3. Optionally save as csv to quick view of the data
'''

from scipy.io import loadmat
import os
import sys
import numpy as np
import pandas as pd

# Expected .mat structure
# file.X = [window x sample]
# file.y = [window x 1]

datasetPath = './datasets/mat'
savePath = './datasets/npz'

visual = False

Xs = [] # All x
ys = [] # All y

if not os.path.exists(datasetPath):
    print(f'{datasetPath} does not exist')
    sys.exit(1)

for file in os.listdir(datasetPath):
    if not file.lower().endswith(".mat"):
        continue
    
    fullpath = os.path.join(datasetPath, file)
    data = loadmat(fullpath)
    filename = file.split('.')[0] #filename.mat -> [filename,mat]
    varname = 'DATA'
    
    X = data[varname]['X'][0,0] # Accessing X (data)
    y = data[varname]['y'][0,0] # Accessing y (label)
    
    y = y.reshape(-1) # Making sure 1D array

    ### IF decide to down-sample X, should happen in here ###
    
    Xs.append(X)
    ys.append(y)

X_all = np.vstack(Xs)          # [total_windows, N_windowSample]
y_all = np.concatenate(ys)     # [total_windows]

print(f'Total X size: {X_all.shape}')
print(f'Total y size: {y_all.shape}')

if not os.path.exists(savePath):
    print(f'{savePath} does not exist')
    sys.exit(1)


saveFilePath = os.path.join(savePath, 'data')
np.savez_compressed(saveFilePath, X=X_all, y=y_all) # Save as .npz file


# Visual check (optional)
if visual:
    data = np.load('./datasets/.npz/data.npz', allow_pickle=True)
    X = data['X']
    y = data['y']
    
    outPath = './datasets/.npz/optional.csv'
    
    df = pd.concat([pd.Series(y, name='label'), pd.DataFrame(X)], axis=1)
    
    df.to_csv(outPath, index=False)
