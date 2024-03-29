"""
Die Funktionen in Fun werden gebraucht, um die Spline auszurechnen und
die Differentialgleichung für die Konstante geschwindigkeit
"""
import numpy as np
import Variables as dat


def spline(stepX, a, b, c, d, N, t):
    """
    :param stepX: Zeitpunkt
    :param a:     Parameter a der Spline
    :param b:     Parameter b der Spline
    :param c:     Parameter c der Spline
    :param d:     Parameter d der Spline
    :param N:     Anzahl parameter
    :param t:     gegebene punkte in der Zeit (wichtig um die parameter a-d zu wechseln)
    :return:      wert der splein zur zeit StepX
    """
    for i in range(N):
        if (stepX <= t[i + 1]):
            return a[i] + b[i] * (stepX - t[i]) + c[i] * pow(stepX - t[i], 2) + d[i] * pow(stepX - t[i], 3)


def spline_num(stepX, a, b, c, d, N, t):
    """ Diese funktio wird im nicht verwendet
    spline_num ist eine abgeänderte version der funktion spline für
     den fall, dass man numerisch differenzieren möchte"""
    return a + b * (stepX - t) + c * pow(stepX - t, 2) + d * pow(stepX - t, 3)


def coeff(x, y):
    """
    :param x: x-koordinaten der gegebenen Punkte
    :param y: y-koordinaten der gegebenen Punkte
    :return:  Koeffizienten der spline a-d und Anzahl Koeffizienten als N
    """
    N = len(y) - 1  # itterationslänge
    a = np.zeros(N)  # a coeff
    b = np.zeros(N)  # b coeff
    c = np.zeros(N)  # c coeff
    d = np.zeros(N)  # d coeff

    h = np.zeros(N)  # Schritte von euler
# --------------------------------Aufstellen der gleichungen als tridiagonalmatrix-----------------
    for i in range(N):
        h[i] = x[i + 1] - x[i]

    A = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        A[i, i] = 2 * (h[i] + h[i + 1])
        if i < N - 2:
            A[i, i + 1] = h[i + 1]
            A[i + 1, i] = h[i + 1]

    f = np.zeros(N - 1)
# -----------------------------Aufstellen der Rechten seite des Gleichungssystems
    for i in range(N - 1):
        f[i] = 6 * (y[i + 2] - y[i + 1]) / h[i + 1] - 6 * (y[i + 1] - y[i]) / h[i]

# Mein eigener TDMA solver nimmt keine Matrizen sondern 3 Vektoren für die linke seite des
# Gleichungssystems und einen Vektor als rechte seite desshalb werden hier die entsprechenden
# Vektoren extrahiert
    a[0] = 0
    c[N - 1] = 0
    for i in range(1, N - 1):
        a[i] = A[i, i - 1]
        c[i] = A[i - 1, i]
    for i in range(N - 1):
        b[i] = A[i, i]
    b[0] = A[0, 0]

    del A # Freigeben des Speichers

    ydd = TDMA_solver(a[:14], b[:14], c[1:15], f)  # Tridiagnonale Matrix lösen
    ydd2 = np.zeros(N + 1)
    # aus der lösung der Tridiagonalmatrix berechnen wir nun die Koeffizienten

    # Lösung modifiziern um Randwerte einzubauen
    ydd2[0] = 0  # Start Randbedingung
    ydd2[N] = 0  # end Randbedingung
    ydd2[1:N]= ydd[:]
    del ydd # Freigeben des speichers

# Koeffiziente ausrechnen
    a = y
    c = ydd2 / 2
    d = (ydd2[1:N + 1] - ydd2[:N]) / (6 * h[:N])
    b = (y[1:N + 1] - y[:N]) / h[:N] - (ydd2[1:N + 1] + 2 * ydd2[:N]) * h[:N] / 6
    return a, b, c, d, N


# calculates the length of a 2D path from to arraya which represent the coordinataes
def PathInt(sx, sy):
    """
    :param sx: x koordinaten des zu integrierenden weges
    :param sy: y koordinaten des zu integrierenden Weges
    :return: Länge des weges
    """
    l = 0
    if len(sx) != len(sy):
        print("error in fun PathInt: Arrays must be the same length")
    for n in range(len(sx) - 2):
        l += np.sqrt(np.power(sx[n + 1] - sx[n], 2) + np.power(sy[n + 1] - sy[n], 2))
    return l


# ableitung des weges in einer Dimension
def s_ds(b, c, d, step_s, t):
    """
    Analytische bleitung zu dem zeitpunkt step_s
    :param b: aktueller b Koeffizient
    :param c: aktueller c Koeffizient
    :param d: aktueller d Koeffizient
    :param step_s: aktueller zeitpunkt
    :param t: zeitliche grnze des aktullen abschnittes
    :return: ableitung im R^1 also nach x ODER y
    """
    b, c, d, step_s, t = map(np.array, (b, c, d, step_s, t))  # copy the array
    x = b + 2 * c * (step_s - t) + d * 3 * (step_s - t) ** 2  # ableitung
    return x


def s_num(a, b, c, d, step_s, t, h):
    """
    Numerische Ableitung wird nicht benötigt
    :param a:  aktueller a Koeff
    :param b:  aktueller b Koeff
    :param c:  aktueller c Koeff
    :param d:  aktueller d Koeff
    :param step_s: aktuelle Zeit
    :param t: grenze des abschnittes
    :param h: schrittgrösse
    :return:  ableitung
    """
    a, b, c, d, step_s, t = map(np.array, (a, b, c, d, step_s, t))  # copy the array
    x1 = float(spline_num(step_s, a, b, c, d, 1, t))
    x2 = float(spline_num(step_s - h, a, b, c, d, 1, t))
    x = (x1 - x2) / h
    return x


# Ableitung (Betrag eines 2D vektors) des weges an einem gegebenen Punkt -> geschwindigkeit
def f(bx_s, cx_s, dx_s, by_s, cy_s, dy_s, y_alt, t_s):
    """
    :param bx_s:  b-Koeffizient am punkt y-alt x-Achse
    :param cx_s:  c-Koeffizient am punkt y-alt x-Achse
    :param dx_s:  d-Koeffizient am punkt y-alt x-Achse
    :param by_s:  b-Koeffizient am punkt y-alt y-Achse
    :param cy_s:  c-Koeffizient am punkt y-alt y-Achse
    :param dy_s:  d-Koeffizient am punkt y-alt y-Achse
    :param y_alt: Zeitpunkt der ableitung
    :param t_s:   Zeitabschnitt
    :return:      2-Norm der Ableitung
    """
    x = (s_ds(bx_s, cx_s, dx_s, y_alt, t_s)) ** 2
    y = (s_ds(by_s, cy_s, dy_s, y_alt, t_s)) ** 2
    g = dat.v_const / np.sqrt(x + y)
    return g


def f_num(ax, bx_s, cx_s, dx_s, ay, by_s, cy_s, dy_s, y_alt, t_s, h):
    x = (s_num(ax, bx_s, cx_s, dx_s, y_alt, t_s, h)) ** 2
    y = (s_num(ay, by_s, cy_s, dy_s, y_alt, t_s, h)) ** 2
    g = dat.v_const / np.sqrt(x + y)
    return g


# ODE nach euler expizit
# Phi(t) hat 15 Abschnitte folglich ein gleichungssystem mit 15 gleichungen
# Phi'(t) = vc/abs(Phi(t)


def explizitEuler(ax, bx, cx, dx, ay, by, cy, dy, t, xend, h, y0, f):
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

    while y[-1] < xend:
        j = int((yalt - np.mod(yalt, 2)) / 2)  # itteration durch koeffizienten
        # explizites Eulerverfahren

        yneu = yalt + h * f(bx[j], cx[j], dx[j], by[j], cy[j], dy[j], yalt, t[j])  # Symbolisch
        # yneu = yalt + h * f_num(ax[j], bx[j], cx[j], dx[j], ay[j], by[j], cy[j], dy[j], yalt, t[j], h) #Nuerisch
        xneu = xalt + h

        # Speichern des Resultats
        y.append(yneu)
        x.append(xneu)

        yalt = yneu
        xalt = xneu
    return np.array(x), np.array(y)


## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMA_solver(a, b, c, d):
    """
    b c 0 0 0 | d
    a b c 0 0 | d
    0 a b c 0 | d
    0 0 a b c | d
    0 0 0 a b | d

    :param a: Untere diagonale
    :param b: mittlere diagonale
    :param c: obere diagonale
    :param d: "Lösung"
    :return: Lösung des systems
    """
    nf = len(a)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy the array
    for it in range(1, nf):
        mc = ac[it] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = ac
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    del bc, cc, dc  # delete variables from memory

    return xc
