import numpy as np


# imports data from a text file in 3 columns
def import_data(Path):
    """
    :param Path:  string-> ort des text-Files
    :return: 3 Spalten des Text files ohne die erste Reihe (separiert in 3 Arrays)
    """
    imported_data = np.loadtxt(Path, skiprows=1)
    t, x, y = imported_data[:, 0], imported_data[:, 1], imported_data[:, 2]
    return x, y, t
