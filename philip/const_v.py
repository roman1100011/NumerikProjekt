import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# --------------------- Funktion zur Berechnung kubischer Splines --------------------------------
def cubic_spline(xraw, yraw, nrofpoints=110):
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

    xcs = np.linspace(xraw[0], xraw[n-1], nrofpoints)  # create points on x to calculate cubic spline y values
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

    return xcs, ycs, b, c, d




# --------------------- Import der Rohdaten --------------------------------------------------------------
def import_data(Path):
    imported_data = np.loadtxt(Path,skiprows=1)
    t, x, y= imported_data[:,0],imported_data[:,1],imported_data[:,2]
    return x, y, t
x, y, t = import_data("/Users/philipkehl/Library/CloudStorage/OneDrive-ZHAW/Semester 4/Numerik/NumerikProjekt/coordinates.txt")


# --------------------- Splineinterpolation berechnen und darstellen -------------------------------------
# Spline rechnen
t_new, x_new, bx, cx, dx = cubic_spline(t, x)
t_new, y_new, by, cy, dy = cubic_spline(t, y)

# # Rohdaten und Interpolation graphisch darstellen
# fig1, ax = plt.subplots()
# ax.set_xlim([t[0] - 1, t[len(t) - 1] + 1])
# ax.set_ylim([min(min(x) - 100, min(y) - 100), max(max(x) + 100, max(y) + 100)])
#
# ax.plot(t, x, 'o', label='x-Rohdaten')
# ax.plot(t, y, 'o', label='x-Rohdaten')
# ax.plot(t_new, x_new, label='x-Spline-Interpolation')
# ax.plot(t_new, y_new, label='y-Spline-Interpolation')
# ax.legend()
# plt.xlabel('t')
# plt.ylabel('x und y')
# plt.show()



# --------------------- Differentialgleichung zur Berechnung eines neuen Zeitvektors tau ---------------------
# Länge einer Spline oder eines Abschnitts davon berechnen
def PathInt(sx, sy):
    l = 0
    if len(sx) != len(sy):
        print("error in fun PathInt: Arrays must be the same length")
    for n in range(len(sx) - 2):
        l += np.sqrt(np.power(sx[n + 1] - sx[n], 2) + np.power(sy[n + 1] - sy[n], 2))
    return l
length = PathInt(x_new, y_new)
print(length)

# Anzahl Elemente innerhalb eines Ranges eines Vektors ausgeben
def count_elements_in_range(vec, lowlim, uplim):
    count = 0
    for el in vec:
        if lowlim < el < uplim:
            count += 1
    return count

v0 = 2 #m/s     Geschwindigkeit



tau0 = 0
for i in range(len(t) -1):                  # Idee: für jeden Abschnitt der Spline soll der tau-Vektor seperat berechnet werden
    def sx(tau):
        return 0 + bx[i]*tau + 2*cx[i]*tau**2 + 3*dx[i]*tau**3

    def sy(tau):
        return 0 + by[i]*tau + 2*cy[i]*tau**2 + 3*dy[i]*tau**3

    def dsx(tau):
        return float(bx[i]) + 2.0*float(cx[i])*tau + 3.0*float(dx[i])*tau**2


    def dsy(tau):
        return by[i] + 2*cy[i]*tau + 3*dy[i]*tau**2

    def f(tau, dsx, dsy):        # DGL rechte Seite
        return v0 * 1.0/np.sqrt(dsx**2 + dsy**2)

    # Länge des aktuellen Splineabschnitts berechnen
    start = count_elements_in_range(t_new, t[0], t[i])
    end =   count_elements_in_range(t_new, t[0], t[i+1])
    length_i = PathInt(x_new[start:end], y_new[start:end])


    # explizites Eulerverfahren zum Lösen der DGL
    def explizitEuler(f, tau0, dsx, dsy, t0, t_end, h=0.1):
        """
        Löst die Differentialgleichung tau'(t) = f(tau(t), dsx(tau), dsy(tau))
        mithilfe des expliziten Euler-Verfahrens.

        Args:
        f: Eine Funktion, die die rechte Seite der Differentialgleichung f(tau, dsx, dsy) berechnet.
        tau0: Der Anfangswert für tau.
        dsx: Eine Funktion, die dsx(tau) berechnet.
        dsy: Eine Funktion, die dsy(tau) berechnet.
        t0: Der Anfangswert für t.
        t_end: Der Endwert für t.
        h: Der Zeitschritt.

        Returns:
        tau: Ein Array mit den Werten für tau an den Zeitpunkten t0, t0+h, t0+2h, ..., t_end.
        """
        num_steps = int((t_end - t0) / h)  # Anzahl der Zeitschritte
        tau = [tau0]  # Anfangswert für tau
        for i in range(num_steps):
            t = t0 + i * h  # Aktueller Zeitpunkt
            tau_i = tau[-1]  # Aktueller Wert für tau
            dsx_i = dsx(tau_i)  # dsx an der Stelle tau_i
            dsy_i = dsy(tau_i)  # dsy an der Stelle tau_i
            tau_new = tau_i + h * f(tau_i, dsx_i, dsy_i)  # Berechne neuen Wert für tau
            tau.append(tau_new)  # Füge neuen Wert für tau hinzu
        return tau

    tau_seq = explizitEuler(f, tau0, dsx, dsy, t[i], t[i+1])
    print(max(tau_seq)-tau0)
    tau0 = max(tau_seq)
    if i == 0:
        tau = tau_seq
    else:
        tau = np.concatenate((tau, tau_seq))

#print(tau)



# --------------------- Prüfen ob Geschwindigkeit konstant ist -------------------------------



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
animationKonstant = FuncAnimation(fig2, updateKonstant, frames=len(t_new), interval=50, repeat=True)

# Anzeigen der Animation
plt.show()