from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


def Animate(x,y,sx,sy,step):
    # Zeitliche Animation der Rohdaten und der Interpolation
    fig2, ax = plt.subplots()
    ax.set_xlim([min(x) - 50, max(x) + 50])
    ax.set_ylim([min(y) - 50, max(y) + 50])
    # Plotten der Punkte
    points, = ax.plot([], [], 'g-', label='Spline')
    ax.plot(x, y, 'ro', label='Rohdaten')
    # Update-Funktion für die Animation
    def update(i):
        # Setzen der neuen Daten für die Punkte
        points.set_data(sx[:i + 1], sy[:i + 1])

        # Rückgabe des geänderten Punkte-Objekts
        return points,


    # Erstellung der Animation
    animation = FuncAnimation(fig2, update, frames=len(step), interval=25, repeat=True)

    # Anzeigen der Animation
    plt.show()
def plot_Phi(phi):
    x = np.linspace(0,len(phi),600)/30
    fig1 , ax1 = plt.subplots()
    ax1.set_xlim([min(x) - 5, max(x) + 5])
    ax1.set_ylim([min(phi) - 10, max(phi) + 10])
    ax1.plot(x, phi, 'ro', label='Phi')
    plt.grid = True
    plt.show()


