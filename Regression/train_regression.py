# train_regression.py
# -------------------
# 1) Leser treningsdata
# 2) Gjør modellvalg med GridSearchCV (PolynomialFeatures + StandardScaler + Ridge)
# 3) Trener beste modell på hele datasettet
# 4) Lagrer modellen (best_model.pkl) + en liten JSON med beste hyperparametre

import json, joblib, numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

import os
print("Current working directory:", os.getcwd())


# --- 1) Data ---
base = Path(__file__).resolve().parent  # path to this script
X = np.load(base / "X_train.npy")   # shape (700, 6)
y = np.load(base / "Y_train.npy")   # shape (700,)

# --- 2) Pipeline + hyperparametre ---
pipe = Pipeline([
    ("poly",   PolynomialFeatures(include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge",  Ridge())
])

param_grid = {
    "poly__degree": [2, 3, 4],                # prøv polynomgrad 2–4
    "ridge__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100],  # regulariseringsstyrke
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="r2", refit=True, n_jobs=-1)

print("Running GridSearchCV (kan ta litt tid)...")
gs.fit(X, y)
print("Best params:", gs.best_params_, " | CV R^2:", gs.best_score_)

# --- 3) Tren beste modell på hele datasettet ---
best_model = gs.best_estimator_
best_model.fit(X, y)

# --- 4) Lagre modell og metadata ---
joblib.dump(best_model, base / "best_model.pkl")
with open(base / "best_model_info.json", "w") as f:
    json.dump({
        "best_params": gs.best_params_,
        "best_cv_r2": gs.best_score_,
        "model_family": "PolynomialFeatures + StandardScaler + Ridge"
    }, f, indent=2)

print("Lagret: best_model.pkl og best_model_info.json")
