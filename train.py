from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import os 
import sys
import json
import joblib

dataPath = './datasets/npz/data_feature.npz'
paramPath = './model/hyperparameter.json'
savePath = './model'

# Load data
if not os.path.exists(dataPath):
    print(f'{dataPath} does not exist')
    sys.exit(1)

data = np.load(dataPath)

X_train = data['X_train']
X_test = data['X_test']

y_train = data['y_train']
y_test = data['y_test']


# Load hyperparameters
if not os.path.exists(paramPath):
    print(f'{paramPath} does not exist')
    sys.exit(1)

with open(paramPath, 'r') as f:
    params = json.load(f)


model = RandomForestClassifier(**params, class_weight='balanced', random_state=42)

model.fit(X_train, y_train)

test_pred = model.predict(X_test)
print(classification_report(y_test, test_pred))

# Save model
if not os.path.exists(savePath):
    print(f'{savePath} does not exist')
    sys.exit(1)

# Save as joblib
saveModelPath = os.path.join(savePath, 'model.joblib')
joblib.dump(model, saveModelPath)