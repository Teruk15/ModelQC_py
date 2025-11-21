from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import optuna
import os 
import sys
import json

dataPath = './datasets/.npz/data_feature.npz'
savePath = './model'

# Load data
if not os.path.exists(dataPath):
    print(f'{dataPath} does not exist')
    sys.exit(1)

data = np.load(dataPath)

X_train = data['X_train']
X_val = data['X_val']

y_train = data['y_train']
y_val = data['y_val']


def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 600, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_float("max_features", 0.1, 1.0)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        n_jobs=-1,          # keep parallelism here
        random_state=42
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scores = cross_val_score(
        clf,
        X_train,
        y_train,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1            # <- IMPORTANT: no parallel CV
    )

    return scores.mean()

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
best_params = study.best_params

print("Best params:")
print(best_params)

print("Best F1 score:", study.best_value)

rf = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)
print(classification_report(y_val, val_pred))

# Save
if not os.path.exists(savePath):
    print(f'{savePath} does not exist')
    sys.exit(1)

saveFilePath = os.path.join(savePath, 'hyperparameter.json')
with open(saveFilePath, 'w') as f:
    json.dump(best_params, f, indent=4)

print(f'Saved hyperparameters at {saveFilePath}')