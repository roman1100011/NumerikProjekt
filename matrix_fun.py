from scipy.linalg import solve_banded
import numpy as np





def tridiagonal_matrix(y, t):
    h = t[1:] - t[:-1]
    a = np.array([])
    b = np.array([])
    c = np.array([])
    d = np.array([])
    for i in range(1, len(t) - 1):
        a = np.append(a, h[i] / (h[i] + h[i + 1]))
        c = np.append(c, 1 - a[-1])
        b = np.append(c, 2)
        d = np.append(d, 6 * ((y[i + 1] - y[i]) / h[i + 1] - (y[i] - y[i - 1]) / h[i]))
    # extract the needed parameters for the tridiagnol matrix
    alpha = a[1:]
    beta = b[1:-1]
    gamma = c[:-1]
    r = d[1:-1]
    return h, alpha, beta, gamma, r


# peace the tridiagonal matrix together as a Banded matrix and solve it using the banded solver from scipy
def solve_tridiagonal_matrix(alpha, beta, gamma, r):
    n = len(alpha)
    ab = np.zeros((2, n))
    ab[0] = beta
    ab[1] = alpha
    ab[1][0] = 0
    ab[0][-1] = 0
    ab[0] = np.insert(ab[0], 0, 0)
    ab[1] = np.append(ab[1], 0)
    ab = np.vstack((gamma, ab, gamma))
    r = np.insert(r, 0, 0)
    r = np.append(r, 0)
    m = solve_banded((1, 1), ab, r)
    return m


# Calculate coefficients
def spli_coeffs(m, h, t, y):
    from main import a_coeffs, b_coeffs, c_coeffs, d_coeffs
    m = m.copy()
    h = h.copy()
    for i in range(len(t) - 1):
        a_coeffs.append((m[i + 1] - m[i]) / (6 * h[i]))
        b_coeffs.append(m[i] / 2)
        c_coeffs.append((y[i + 1] - y[i]) / h[i] - (2 * h[i] * m[i] + h[i] * m[i + 1]) / 6)
        d_coeffs.append(y[i])
    return ()


# generate spline
def make_spline(t,):
    from main import a_coeffs, b_coeffs, c_coeffs, d_coeffs
    from graphics import polt_still
    num_points = 100
    spline_x = []
    spline_y = []
    for i in range(len(t) - 1):
        for j in range(num_points):
            t_val = t[i] + j * (t[i + 1] - t[i]) / num_points
            delta_t = t_val - t[i]
            x_val = a_coeffs[i] * delta_t ** 3 + b_coeffs[i] * delta_t ** 2 + c_coeffs[i] * delta_t + d_coeffs[i]
            y_val = a_coeffs[i] * delta_t ** 3 + b_coeffs[i] * delta_t ** 2 + c_coeffs[i] * delta_t + d_coeffs[i]
            spline_x.append(x_val)
            spline_y.append(y_val)
    polt_still(spline_x, spline_y)

