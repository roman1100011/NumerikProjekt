import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matrix_fun
import graphics
import data_handling
#------------------------Daten aus Text file einlesen--------------------------
x,y,t = data_handling.import_data("data/coordinates.txt")
x = np.array((x))
y = np.array((y))
t = np.array((t))
points = [[x],[y],[t]]
#--------------------------------display raw data---------------------------------------------------
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
#-----------------------Berechnung der X-Koordinaten--------------------------

#-----------------------spline interpolationsmatrix in der XY-ebebne---------

#-------------------------Berechnung der geschwindigkeit und Beschleunigung---------------

#------------------------------Konstante geschwindigkeit---------------------

