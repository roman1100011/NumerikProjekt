import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matrix_fun
import graphics
import data_handling
# --------------------------Global Variables----------------------------------
h = np.array([]) # distance between points in time

# coefficients for spline
a_coeffs = []
b_coeffs = []
c_coeffs = []
d_coeffs = []

alpha = np.array([])
beta = np.array([])
gamma = np.array([])
r = np.array([])
# ------------------------Daten aus Text file einlesen--------------------------
x, y, t = data_handling.import_data("data/coordinates.txt")
x = np.array(x)
y = np.array(y)
t = np.array(t)

points = [[x], [y], [t]]
# --------------------------------display raw data---------------------------------------------------
# create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim([0, 600])
ax.set_ylim([0, 800])


# create the line plot
line, = ax.plot(x[:1], y[:1])
# set the linewidth
line.set_linewidth(4)
# update function for the animation


def update(frame):
    # update the data of the line plot
    line.set_data(x[:frame+1], y[:frame+1])
    return line,


# create the animation
animation = FuncAnimation(fig, update, frames=len(t), interval=300)

# show the animation
plt.show()
# -----------------------Berechnung der X-Koordinaten--------------------------
# calculate the distance between points in time
h, alpha, beta, gamma, r = matrix_fun.tridiagonal_matrix(y, t)

m = matrix_fun.solve_tridiagonal_matrix(alpha, beta, gamma, r)

matrix_fun.spli_coeffs(m, h)

# -----------------------spline interpolations matrix in der XY-Ebebne---------

# -------------------------Berechnung der geschwindigkeit und Beschleunigung---------------

# ------------------------------Konstante geschwindigkeit---------------------

