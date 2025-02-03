import numpy as np
import pandas as pd


def rolling_average(a, n):
    """Pandas rolling average"""
    return pd.Series(a).rolling(n, center=True, min_periods=1).mean().to_numpy()


def find_closest_node(node, nodes):
    """Find the index of the closest node to the given node"""
    return np.argmin(np.sum((np.asarray(nodes) - node)**2, axis=1))


def find_nearest(array, value):
    """Find the closest point to a value in an array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

