"""
Main: Genereller Programmablauf; keine definition von Funktionen

data_handling: Funktionen für den Import und formatieren von gegebenen daten

animation: Funktionen zur darstellung und animation von daten

Variables: Einige Daten wie die länge der Spline und geschwindigkeit müssen nicht immer von neuem ausgerechnet werden.
            Desshalb sind solche variabeln in diesem file festgehalten
"""
from typing import Any

import numpy as np

import Fun
import Variables as dat
import animation
from Fun import coeff, spline
from animation import Animate
from data_handling import import_data

# Daten Importieren  -> Achtung lokaler Path!!!!!!
x, y, t = import_data("/Users/romanberti/PycharmProjects/scientificProject/coordinates.txt")

# Ausrechnen der koefizienten aus den gegebenen Punkten
bx: float | Any
ax, bx, cx, dx, N = coeff(t, x)
ay, by, cy, dy, N = coeff(t, y)

# -----------------------------------------------------------------------------------------------------------
# Spline vorbereiten /arrays werden definiert
dt = 0.05
step = np.arange(t[0], t[N] + dt, dt)
sxv = np.zeros(len(step))  # Array für weg-punkte mit variabler geschwindigkeit x koordinaten
syv = np.zeros(len(step))  # Array für wegpunkte mit variabler geschwindigkeit y koordinaten

# Spline ausrechnen aus Koeffizienten -> ploten
for i in range(len(step)):
    sxv[i] = spline(step[i], ax, bx, cx, dx, N, t)  # Aus den Koeffizienten die datenpunkte für die Spline
    syv[i] = spline(step[i], ay, by, cy, dy, N, t)  # mit variabler geschwindigkeit ausrechnen

# Diff gleichung mit euler expizit lösen
step_new, phi = Fun.explizitEuler(ax, bx, cx, dx, ay, by, cy, dy, t, np.max(t), 0.05, dat.y0, Fun.f)
animation.plot_Phi(phi, step)

# Spline Vorbereiten für konstant Geschwindigkeit
sxc = np.zeros(len(phi))  # Array für weg-punkte mit konstanter geschwindigkeit x koordinaten
syc = np.zeros(len(phi))  # Array für weg-punkte mit konstanter geschwindigkeit y koordinaten

for i in range(len(phi)):
    sxc[i] = spline(phi[i], ax, bx, cx, dx, N, t)  # Aus den Koeffizienten und phi die datenpunkte für die Spline
    syc[i] = spline(phi[i], ay, by, cy, dy, N, t)  # mit konstanter geschwindigkeit ausrechnen

# Animation der Punkte und Spline
Animate(x, y, sxv, syv, sxc, syc, step, phi)
