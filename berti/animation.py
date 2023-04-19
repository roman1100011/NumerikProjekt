from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np



def Animate(x, y, sxv, syv,sxc,syc, step, phi):
    # Zeitliche Animation der Rohdaten und der Interpolation
    """

    :param x:   x-koordinate der gegebenen punkte
    :param y:   y-koordinate der gegebenen punkte
    :param sxv: x-koordinate, der berechneten spline mit variabler geschwindigkeit
    :param syv: y-koordinate, der berechneten spline mit variabler geschwindigkeit
    :param sxc: x-koordinate, der berechneten spline mit gleicher geschwindigkeit
    :param syc: y-koordinate, der berechneten spline mit gleicher geschwindigkeit
    :param step: zeitvektor der uhrsprünglichen spline (die daten werden nicht direkt gebraucht, nur die länge)
    :param phi:  zeitvektor der korrigierter spline (die daten werden nicht direkt gebraucht, nur die länge)
    :return: keinen wert
    """
    #durch die teils grösseren schrittgrösen ist jewils ein vektor etwas länger deshalb muss das korrigiert werden

    k = -1                                               # temporary variable to compare the length of the arrays
    len_array = [len(sxc),len(sxv)]                      # array which contains the array length of the two splines (Assumption: the x and y vectors in the same spline are the same length)
    max_len_ind = len_array.index(max(len_array))

    if max_len_ind == 0:                                 # Verlängern der Vektoren durch repetieren des letzte eintrages
        for i in range(len(sxv)-1,len(sxc)-1):
            sxv = np.append(sxv, sxv[i])
            syv = np.append(syv, syv[i])
    if max_len_ind == 1:
        for i in range(len(sxc)-1,len(sxv)-1):
            sxc = np.append(sxc, sxc[i-1])
            syc = np.append(syc, syc[i-1])


    #settup des Plotes
    fig2, ax = plt.subplots()
    ax.set_xlim([min(x) - 50, max(x) + 50])
    ax.set_ylim([min(y) - 50, max(y) + 50])
    # Plotten der Punkte


    points, = ax.plot([1,1],[0,0] , "o", label='point_1') # Startpunkte darstellen
    ax.plot(x, y, 'ro', label='Rohdaten')         # Gegebene punkt darstellen
    ax.plot(sxv, syv)                             # Berechnete spline darstellen


    # Update-Funktion für die Animation
    def update(i):
        # Setzen der neuen Daten für die Punkte
        points.set_data([sxv[i + 1],sxc[i+1]], [syv[i + 1],syc[i+1]])

        # Rückgabe des geänderten Punkte-Objekts
        return points,


    # Erstellung der Animation
    animation = FuncAnimation(fig2, update, frames=len(step), interval=25, repeat=True)

    # Anzeigen der Animation
    plt.show()
def plot_Phi(phi,step):
    x = np.linspace(0,len(phi),600)/30
    fig1 , ax1 = plt.subplots()
    ax1.set_xlim([0 - 5, max(step) + 5])
    ax1.set_ylim([min(phi) - 10, max(phi) + 10])
    ax1.plot(step, phi, 'ro', label='Phi')
    plt.grid = True
    plt.show()


def plot_Phi2(phi,step):
    x = np.linspace(0,len(phi),600)/30
    fig1 , ax1 = plt.subplots()
    ax1.set_xlim([min(x) - 5, max(x) + 5])
    ax1.set_ylim([min(phi) - 10, max(phi) + 10])
    ax1.plot(x, phi[:-4], 'ro', label='Phi')
    ax1.plot(x,step[:-1])
    plt.grid = True
    plt.show()