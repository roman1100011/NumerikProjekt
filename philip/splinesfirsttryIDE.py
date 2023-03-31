import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


# Definiere eine Funktion zur Berechnung von kubischen Splines
def cubic_spline(xraw, yraw):
    n = len(xraw)
    h = np.diff(xraw)
    alpha = np.zeros(n - 1)
    for i in range(1, n - 1):
        alpha[i] = 3 / h[i] * (yraw[i + 1] - yraw[i]) - 3 / h[i - 1] * (yraw[i] - yraw[i - 1])

    l = np.zeros(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    l[0] = 1
    mu[0] = z[0] = 0

    for i in range(1, n - 1):
        l[i] = 2 * (xraw[i + 1] - xraw[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n - 1] = 1
    z[n - 1] = 0
    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    xcs = np.linspace(xraw[0], xraw[-1], 100)  # create 100 points on x to calculate cubic spline y values
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (yraw[j + 1] - yraw[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    ycs = np.zeros(len(xcs))
    for i in range(len(xcs)):
        for j in range(len(xraw) - 1):
            if xraw[j] <= xcs[i] <= xraw[j + 1]:
                ycs[i] = yraw[j] + b[j] * (xcs[i] - t[j]) + c[j] * (xcs[i] - t[j]) ** 2 + d[j] * (xcs[i] - t[j]) ** 3
                break

    return xcs, ycs


# Input der Rohdaten
t = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
x = np.array([0, 400, 500, 500, 400, 50, 50, 200, 300, 460, 440, 150])
y = np.array([0, 0, 100, 400, 500, 500, 575, 700, 700, 575, 400, 65])

# Splineinterpolation mit der erstellten Funktion berechnen
t_new, x_new = cubic_spline(t, x)
t_new, y_new = cubic_spline(t, y)

"""
# Rohdaten und Interpolation graphisch darstellen
fig, ax = plt.subplots()
ax.set_xlim([t[0] - 1, t[len(t) - 1] + 1])
ax.set_ylim([min(min(x) - 1, min(y) - 1), max(max(x) + 1, max(y) + 1)])

ax.plot(t, x, 'o', label='x-Rohdaten')
ax.plot(t, y, 'o', label='x-Rohdaten')
ax.plot(t_new, x_new, label='x-Spline-Interpolation')
ax.plot(t_new, y_new, label='y-Spline-Interpolation')
ax.legend()
plt.show()
"""


# Zeitliche Animation der Rohdaten und der Interpolation
fig2, ax = plt.subplots()
ax.set_xlim([min(x) - 50, max(x) + 50])
ax.set_ylim([min(y) - 50, max(y) + 50])

# Plotten der Punkte
points, = ax.plot([], [], 'b-', label='Spline')
ax.plot(x, y, 'ro', label='Rohdaten')


# Update-Funktion für die Animation
def update(i):
    # Setzen der neuen Daten für die Punkte
    points.set_data(x_new[:i + 1], y_new[:i + 1])

    # Rückgabe des geänderten Punkte-Objekts
    return points,


# Erstellung der Animation
animation = FuncAnimation(fig2, update, frames=len(t_new), interval=25, repeat=False)

# Anzeigen der Animation
plt.show()