import numpy as np

#imports data from a text file in 3 colums
def import_data(Path):
    imported_data = np.loadtxt(Path,skiprows=1)
    t, x, y= imported_data[:,0],imported_data[:,1],imported_data[:,2]
    return x, y, t