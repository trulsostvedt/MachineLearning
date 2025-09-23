# mymodel.py
import os
import numpy as np
import joblib

_model = None

def _load_model():
    global _model
    if _model is None:
        here = os.path.dirname(__file__)
        path = os.path.join(here, "best_model.pkl")
        _model = joblib.load(path)
    return _model

def predict(Xtest):
    """
    Parameters
    ----------
    Xtest : array-like of shape (n_samples, 6)
        Test features

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Predictions
    """
    model = _load_model()
    Xtest = np.asarray(Xtest)
    y_pred = model.predict(Xtest)
    return np.asarray(y_pred).reshape(-1,)
