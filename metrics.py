from functools import partial
import numpy as np


def _within_x(y_true, y_pred, x):
    """
    Calculate the percentage of predictions that are within x% of the true value.
    """
    return (np.abs(y_true - y_pred) < x * y_true).sum() / len(y_true)


within_1percent  = partial(_within_x, x=0.01)
within_5percent  = partial(_within_x, x=0.05)
within_10percent = partial(_within_x, x=0.10)
within_20percent = partial(_within_x, x=0.20)
within_50percent = partial(_within_x, x=0.50)
