import numpy as np
from tdmSolve import TDMA_solver
import Variables as dat


def spline(stepX, a, b, c, d, N, t):
    if (stepX < t[0]) or (stepX > t[N]):
        print("grenze überschritten")
        return
    for i in range(N):
        if (stepX <= t[i + 1]):
            return a[i] + b[i] * (stepX - t[i]) + c[i] * pow(stepX - t[i], 2) + d[i] * pow(stepX - t[i], 3)


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

# Gesuchte funktion /aktuell nicht in verwendung
def singelPhi(m, step_s):
    return m * step_s

# Ableitung (Betrag eines 2D vektors) des weges an einem gegebenen Punkt -> geschwindigkeit
def f(step_s, bx_s, cx_s, dx_s, by_s, cy_s, dy_s, t_s):
    x = (s(bx_s, cx_s, dx_s, step_s, t_s)) ** 2
    y = (s(by_s, cy_s, dy_s, step_s, t_s)) ** 2
    g = dat.v_const / np.sqrt(x + y)
    return g


# ODE nach euler expizit
# Phi(t) hat 15 Abschnitte folglich ein gleichungssystem mit 15 gleichungen
# Phi'(t) = vc/abs(Phi(t)
def solveEulerex(step, vc, ax, bx, cx, dx, ay, by, cy, dy, t):
    dh = 0.05 # Schrittgrösse aktuell
    h = np.arange(t[0] + dh, t[-1] + dh, dh)
    Phi_m = np.zeros(len(h))
    Phi_m[0] = dat.y0 # Startwert für eulerverfahren

    for i in range(1, len(h) - 1):
        j = int((h[i] - np.mod(h[i], 2)) / 2) # itteration durch die Koeffizienten
        Phi_m_prev = Phi_m[i - 1]
        t_curr = h[i]
        fcurent = f(Phi_m_prev, bx[j], cx[j], dx[j], by[j], cy[j], dy[j], t_curr)
        Phi_m[i] = Phi_m_prev + dh * fcurent  # Vorwärtsschritt

    return Phi_m
