import numpy as np
from tdmSolve import TDMA_solver
import Variables as dat


def spline(stepX, a, b, c, d, N, t):
    #if (stepX < t[0]) or (stepX > t[N]):
     #   print("grenze überschritten")
        #return
    for i in range(N):
        if (stepX <= t[i + 1]):
            return a[i] + b[i] * (stepX - t[i]) + c[i] * pow(stepX - t[i], 2) + d[i] * pow(stepX - t[i], 3)
def spline_num(stepX, a, b, c, d, N, t):
    return a + b * (stepX - t) + c * pow(stepX - t, 2) + d * pow(stepX - t, 3)


def coeff(x, y):
    N = len(y) - 1  # itterationslänge
    a = np.zeros(N)  # a coeff
    b = np.zeros(N)  # b coeff
    c = np.zeros(N)  # c coeff
    d = np.zeros(N)  # d coeff

    h = np.zeros(N)  # Schritte von euler

    for i in range(N):
        h[i] = x[i + 1] - x[i]

    A = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        A[i, i] = 2 * (h[i] + h[i + 1])
        if i < N - 2:
            A[i, i + 1] = h[i + 1]
            A[i + 1, i] = h[i + 1]

    f = np.zeros(N - 1)
    # f''(x_i)       = (f(x_{i-1}) - 2f(x_i) + f(x_{i + 1})) / h_i ^ 2
    # f''(x_{i + 1}) = (f(x_i) - 2f(x_{i+1}) + f(x_{i + 2})) / h_{i + 1} ^ 2
    for i in range(N - 1):
        f[i] = 6 * (y[i + 2] - y[i + 1]) / h[i + 1] - 6 * (y[i + 1] - y[i]) / h[i]
    # Einschub test eigener Solver aus der matrix die gesuchten koeffizienten herauslesen

    a[0] = 0
    c[N - 1] = 0
    for i in range(1, N - 1):
        a[i] = A[i, i - 1]
        c[i] = A[i - 1, i]
    for i in range(N - 1):
        b[i] = A[i, i]
    b[0] = A[0, 0]

    ydd = TDMA_solver(a[:14], b[:14], c[1:15], f) # Tridiagnonale Matrix lösen
    ydd2 = np.zeros(N + 1)
    ydd2[0] = 0  # Start Randbedingung
    ydd2[N] = 0  # end Randbedingung
    for i in range(N - 1):
        ydd2[i + 1] = ydd[i]

    a = y
    c = ydd2 / 2
    d = (ydd2[1:N + 1] - ydd2[:N]) / (6 * h[:N])
    b = (y[1:N + 1] - y[:N]) / h[:N] - (ydd2[1:N + 1] + 2 * ydd2[:N]) * h[:N] / 6
    return a, b, c, d, N


# calculates the length of a 2D path from to arraya which represent the coordinataes
def PathInt(sx, sy):
    l = 0
    if len(sx) != len(sy):
        print("error in fun PathInt: Arrays must be the same length")
    for n in range(len(sx) - 2):
        l += np.sqrt(np.power(sx[n + 1] - sx[n], 2) + np.power(sy[n + 1] - sy[n], 2))
    return l

# ableitung des weges in einer Dimension
def s(b,c,d,step_s,t):
    b,c,d,step_s,t = map(np.array, (b,c,d,step_s,t))  # copy the array
    x = b+2*c*(step_s-t)+d*3*(step_s-t)**3 # ableitung
    return x


def s_num(a,b,c,d,step_s,t,h):
    """
    :param a:  aktueller a Koeff
    :param b:  aktueller b Koeff
    :param c:  aktueller c Koeff
    :param d:  aktueller d Koeff
    :param step_s: aktuelle zeit
    :param t: grenze des abschnittes
    :param h: schrittgrösse
    :return:  ableitung
    """
    a, b, c, d, step_s, t = map(np.array, (a, b,c,d,step_s,t))  # copy the array
    x1 = float (spline_num(step_s, a, b, c, d, 1, t))
    x2 = float (spline_num(step_s-h , a, b, c, d, 1, t))
    x = (x1-x2)/h
    return x

# Gesuchte funktion /aktuell nicht in verwendung
def singelPhi(m, step_s):
    return m * step_s

# Ableitung (Betrag eines 2D vektors) des weges an einem gegebenen Punkt -> geschwindigkeit
def f( bx_s, cx_s, dx_s, by_s, cy_s, dy_s, y_alt, t_s):
    x = (s(bx_s, cx_s, dx_s, y_alt, t_s)) ** 2
    y = (s(by_s, cy_s, dy_s, y_alt, t_s)) ** 2
    g = dat.v_const / np.sqrt(x + y)
    return g


def f_num(ax, bx_s, cx_s, dx_s, ay, by_s, cy_s, dy_s, y_alt, t_s,h):
    x = (s_num(ax, bx_s, cx_s, dx_s, y_alt, t_s, h)) ** 2
    y = (s_num(ay, by_s, cy_s, dy_s, y_alt, t_s, h)) ** 2
    g = dat.v_const / np.sqrt(x + y)
    return g


# ODE nach euler expizit
# Phi(t) hat 15 Abschnitte folglich ein gleichungssystem mit 15 gleichungen
# Phi'(t) = vc/abs(Phi(t)



# TODO: implementation ist nicht so falsch jedoch wird die ableitung an den stellen, bei denen die Parameter geändert werden hoch und daher fehlerhat, ein lösungsansatz ist, das euleverfahren stükweise zu machendamit innerhalb des eulerverfahrens keine "sprungstellen" entstehen
def explizitEuler(ax, bx, cx, dx, ay, by, cy, dy,t, xend, h, y0, f):
    """
    :param bx:  bx koeffizient von Spline (Array)
    :param cx:  cx koeffizient von Spline (Array)
    :param dx:  dx koeffizient von Spline (Array)
    :param by:  by koeffizient von Spline (Array)
    :param cy:  cy koeffizient von Spline (Array)
    :param dy:  dy koeffizient von Spline (Array)
    :param t:   zeit-"stüzpunkte" gegeben durch anfängliche punkte
    :param xend: grösster zeitwert
    :param h:   Schrittgrösse für euler-verfahren
    :param y0:  Anfangswert (Randwert)
    :param f:   funktion
    :return: t, Phi(t)
    """
    x = [0.]
    y = [y0]
    xalt = 0
    yalt = y0

    while y[-1] < xend-h/2:
        j = int((yalt - np.mod(yalt , 2)) / 2) # itteration durch koeffizienten
        # explizites Eulerverfahren
        if j == 15:
            j =14
        yneu = yalt + h*f( bx[j], cx[j], dx[j], by[j], cy[j], dy[j],yalt, t[j]) # Symbolisch
        if int((yneu - np.mod(yneu-h , 2)) / 2) > j:
            if j == 15:
                j = 13
            yneu = yalt + h * f(bx[j+1], cx[j+1], dx[j+1], by[j+1], cy[j+1], dy[j+1], yalt, t[j+1])  # Symbolisch

        xneu = xalt + h

        # Speichern des Resultats
        y.append(yneu)
        x.append(xneu)

        yalt = yneu
        xalt = xneu
    return np.array(x), np.array(y)
