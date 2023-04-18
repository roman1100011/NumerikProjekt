import numpy as np
import matplotlib.pyplot as pylab

import Fun
import animation
from Fun import coeff, Spline
from data_handling import import_data
from animation import Animate
import Variables as dat



#Daten Importieren
x, y, t = import_data("/Users/romanberti/PycharmProjects/sandbox/coordinates.txt")

#Calculate coeffitients
ax, bx, cx, dx, N = coeff(t, x)
ay, by, cy, dy, N = coeff(t, y)

#-----------------------------------------------------------------------------------------------------------
#Spline vorbereiten
dt = 0.05
step = np. arange(t[0], t[N] + dt, dt)
sx = np.zeros(len(step))
sy = np.zeros(len(step))



#Spline ausrechnen aus koefizienten -> ploten
for i in range(len(step)):
    sx[i] = Spline(step[i], ax, bx, cx, dx, N,t)
    sy[i] = Spline(step[i], ay, by, cy, dy, N,t)


phi = Fun.solveEulerex(step,dat.v_const,ax,bx,cx,dx,ay,by,cy,dy,t)
animation.plot_Phi(phi)

#Animiation der Punkte und Spline
Animate(x,y,sx,sy,step)
