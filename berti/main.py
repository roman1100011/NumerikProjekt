from typing import Any

import numpy as np
import matplotlib.pyplot as pylab

import Fun
import animation
from Fun import coeff, spline
from data_handling import import_data
from animation import Animate
import Variables as dat



#Daten Importieren  -> Achtung lokaler Path
x, y, t = import_data("/Users/romanberti/PycharmProjects/scientificProject/coordinates.txt")

#Calculate coeffitients
bx: float | Any
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
    sx[i] = spline(step[i], ax, bx, cx, dx, N, t)
    sy[i] = spline(step[i], ay, by, cy, dy, N, t)

# Constant speed
# phi = Fun.solveEulerex(step,dat.v_const,ax,bx,cx,dx,ay,by,cy,dy,t)
step_new, phi = Fun.explizitEuler(ax, bx,cx,dx, ay, by,cy,dy,t,np.max(t),0.05,dat.y0,Fun.f)
animation.plot_Phi2(phi,step)


#Delet privios Path
del sy
del sx
#Creat again with the correct length
sx = np.zeros(len(phi))
sy = np.zeros(len(phi))


for i in range(len(phi)):
    sx[i] = spline(phi[i], ax, bx, cx, dx, N, t)
    sy[i] = spline(phi[i], ay, by, cy, dy, N, t)


#Animiation der Punkte und Spline
Animate(x,y,sx,sy,phi)
