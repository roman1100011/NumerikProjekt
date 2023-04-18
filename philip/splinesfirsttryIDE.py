import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


# --------------------- Funktion zur Berechnung kubischer Splines --------------------------------
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

    xcs = np.linspace(xraw[0], xraw[-1], 110)  # create points on x to calculate cubic spline y values
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

    return xcs, ycs#, b, c, d




# --------------------- Import der Rohdaten --------------------------------------------------------------
def import_data(Path):
    imported_data = np.loadtxt(Path,skiprows=1)
    t, x, y= imported_data[:,0],imported_data[:,1],imported_data[:,2]
    return x, y, t
x, y, t = import_data("/Users/philipkehl/Library/CloudStorage/OneDrive-ZHAW/Semester 4/Numerik/NumerikProjekt/coordinates.txt")





# --------------------- Splineinterpolation berechnen und darstellen -------------------------------------
# Spline rechnen
t_new, x_new = cubic_spline(t, x)
t_new, y_new = cubic_spline(t, y)

# Rohdaten und Interpolation graphisch darstellen
fig1, ax = plt.subplots()
ax.set_xlim([t[0] - 1, t[len(t) - 1] + 1])
ax.set_ylim([min(min(x) - 100, min(y) - 100), max(max(x) + 100, max(y) + 100)])

ax.plot(t, x, 'o', label='x-Rohdaten')
ax.plot(t, y, 'o', label='x-Rohdaten')
ax.plot(t_new, x_new, label='x-Spline-Interpolation')
ax.plot(t_new, y_new, label='y-Spline-Interpolation')
ax.legend()
plt.xlabel('t')
plt.ylabel('x und y')
plt.show()



# --------------------- Differentialgleichung zur Berechnung eines neuen Zeitvektors tau ---------------------

v0 = 2 #m/s     Geschwindigkeit
s0 = np.array([x_new[0], y_new[0]])

def s():        # Ortsfunktion, x und y Anteil
    return 0


def f(v0, ds):        # DGL rechte Seite
    return v0 * 1/np.sqrt(ds[0]**2 + ds[1]**2)

# explizites Eulerverfahren zum Lösen der DGL
def explizitEuler(xend, y0, f, h=0.01 ):
    x = [0.]
    y = [y0]
    xalt = 0
    yalt = y0
    while x[-1] < xend-h/2:
        # explizites Eulerverfahren
        yneu = yalt + h*f(xalt, yalt)
        xneu = xalt + h
        # Speichern des Resultats
        y.append(yneu)
        x.append(xneu)
        yalt = yneu
        xalt = xneu
    return np.array(x), np.array(y)

te, se = explizitEuler(max(t_new), s0, f)




# --------------------- Zeitliche Animation der Rohdaten und der Interpolation -------------------------------
fig2, ax = plt.subplots()
ax.set_xlim([min(x) - 50, max(x) + 50])
ax.set_ylim([min(y) - 50, max(y) + 50])

# Plotten der Punkte
points, = ax.plot([], [], 'go', label='Spline')
pointsKonstant, = ax.plot([], [], 'yx', label='Spline')
ax.plot(x, y, 'ro', label='Rohdaten')
ax.plot(x_new, y_new, 'b-', label='Spline')
plt.xlabel('x')
plt.ylabel('y')

# Update-Funktion für die Animation
def update(i):
    # Setzen der neuen Daten für die Punkte
    points.set_data(x_new[:i + 1], y_new[:i + 1])

    # Rückgabe des geänderten Punkte-Objekts
    return points,

# Update-Funktion für die Animation mit KONSTANTER geschwindigkeit
def updateKonstant(i):
    # Setzen der neuen Daten für die Punkte
    pointsKonstant.set_data(x_new[:i + 1], y_new[:i + 1])

    # Rückgabe des geänderten Punkte-Objekts
    return pointsKonstant,


# Erstellung der Animation
animation = FuncAnimation(fig2, update, frames=len(t_new), interval=50, repeat=True)
animationKonstant = FuncAnimation(fig2, updateKonstant, frames=len(t_new), interval=50, repeat=False)

# Anzeigen der Animation
plt.show()