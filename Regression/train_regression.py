# Tests 3 model families (Poly+Ridge, Poly+Lasso, KernelRidge RBF).
# Saves best_model.pkl (the winner), best_model_info.json (structured),
# and cv_results.txt (I added this log to use when writing the report, not necessarily for the model).

import json, joblib, numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge

# --- Load data ---
base = Path(__file__).resolve().parent
X = np.load(base / "X_train.npy")
y = np.load(base / "Y_train.npy")

cv = KFold(n_splits=5, shuffle=True, random_state=42) 
# 42 = the answer to life, the universe, and everything - ref to Hitchhiker's Guide to the Galaxy our inside joke
# when choosing random states :) its "random" but fixed so results are reproducible
results = {}

def summarize_gs(gs):
    """Pick out best score/params + std for the best row."""
    idx = gs.best_index_
    mean = float(gs.cv_results_["mean_test_score"][idx])
    std  = float(gs.cv_results_["std_test_score"][idx])
    return {
        "best_params": gs.best_params_,
        "cv_r2_mean": mean,
        "cv_r2_std": std,
    }

# Model 1: Poly+Ridge 
pipe_ridge = Pipeline([
    ("poly",   PolynomialFeatures(include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge",  Ridge())
])
grid_ridge = {
    "poly__degree": [2, 3, 4], # 4 and 5 were very close, but 4 is likely better to avoid overfitting
    "ridge__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100],
}
gs_ridge = GridSearchCV(pipe_ridge, grid_ridge, cv=cv, scoring="r2", refit=True, n_jobs=-1)
gs_ridge.fit(X, y)
results["poly_ridge"] = summarize_gs(gs_ridge)

# Model 2: Poly+Lasso 
pipe_lasso = Pipeline([
    ("poly",   PolynomialFeatures(include_bias=False)),
    ("scaler", StandardScaler()),
    ("lasso",  Lasso(max_iter=50000, tol=1e-3, selection="cyclic")) # Max iter increased to ensure convergence (It didn't converge with default 1000)
])
grid_lasso = {
    "poly__degree": [2, 3, 4], # 4 and 5 were very close, but 4 is likely better to avoid overfitting (see notes.ipynb)
    "lasso__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100], 
}
gs_lasso = GridSearchCV(pipe_lasso, grid_lasso, cv=cv, scoring="r2", refit=True, n_jobs=-1)
gs_lasso.fit(X, y)
results["poly_lasso"] = summarize_gs(gs_lasso)

# Model 3: Kernel Ridge (RBF)
pipe_kr = Pipeline([
    ("scaler", StandardScaler()),
    ("kr",     KernelRidge(kernel="rbf"))
])
grid_kr = {
    "kr__alpha": [0.01, 0.1, 1.0, 10.0],
    "kr__gamma": [1e-3, 1e-2, 1e-1, 1.0],
}
gs_kr = GridSearchCV(pipe_kr, grid_kr, cv=cv, scoring="r2", refit=True, n_jobs=-1)
gs_kr.fit(X, y)
results["kernel_ridge_rbf"] = summarize_gs(gs_kr)

# We select the best model based on CV R^2
best_name = max(results.keys(), key=lambda k: results[k]["cv_r2_mean"])
best_est  = {
    "poly_ridge": gs_ridge.best_estimator_,
    "poly_lasso": gs_lasso.best_estimator_,
    "kernel_ridge_rbf": gs_kr.best_estimator_,
}[best_name]

best_est.fit(X, y)
joblib.dump(best_est, base / "best_model.pkl")

# JSON log; source of inspiration: https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
with open(base / "best_model_info.json", "w") as f:
    json.dump({
        "winner": best_name,
        "results": results,
        "notes": "cv_r2_mean ± cv_r2_std with 5-fold KFold(shuffle, rs=42). "
                 "best_model.pkl is trained on all 700."
    }, f, indent=2)

# Text log for easy reading when we are writing the report
log_path = base / "cv_results.txt"
with open(log_path, "w") as f:
    f.write("CV Results (5-fold, R^2)\n")
    for k, info in results.items():
        mean = info["cv_r2_mean"]; std = info["cv_r2_std"]
        f.write(f"{k:>18s}: {mean:.4f} ± {std:.4f} | params: {info['best_params']}\n")
    f.write(f"\nWinner: {best_name}\n")

print(f"\nWinner: {best_name} --> saved to best_model.pkl")
print(f"Details saved to best_model_info.json and cv_results.txt")


