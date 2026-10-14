"""Binary ACC/AUC/EER definitions used in the manuscript (fractional outputs)."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def binary_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Nonempty, equally sized one-dimensional arrays are required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError('Invalid label or nonfinite probability')
    if ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be in [0, 1]')
    acc = float(accuracy_score(y, p >= 0.5))
    if len(np.unique(y)) < 2:
        return {'acc': acc, 'auc': None, 'eer': None}
    fpr, tpr, _ = roc_curve(y, p, pos_label=1, drop_intermediate=False)
    difference = fpr - (1 - tpr)
    hi = int(np.flatnonzero(difference >= 0)[0])
    lo = hi if difference[hi] == 0 else hi - 1
    weight = 0 if lo == hi else -difference[lo] / (difference[hi] - difference[lo])
    eer = float(fpr[lo] + weight * (fpr[hi] - fpr[lo]))
    return {'acc': acc, 'auc': float(roc_auc_score(y, p)), 'eer': eer}
