import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation



def Animate(x, y, sxv, syv, sxc, syc, step, phi):
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
    # durch die teils grösseren schrittgrösen ist jewils ein vektor etwas länger deshalb muss das korrigiert werden


    len_array = [len(sxc),len(sxv)]  # array which contains the array length of the two splines (Assumption: the x and y vectors in the same spline are the same length)
    max_len_ind = len_array.index(max(len_array))

    if max_len_ind == 0:  # Verlängern der Vektoren durch repetieren des letzte eintrages
        for i in range(len(sxv) - 1, len(sxc) - 1):
            sxv = np.append(sxv, sxv[i])
            syv = np.append(syv, syv[i])
            step = np.append(step, step[i])
    if max_len_ind == 1:
        for i in range(len(sxc) - 1, len(sxv) - 1):
            sxc = np.append(sxc, sxc[i - 1])
            syc = np.append(syc, syc[i - 1])
            phi = np.append(phi, phi[i])

    # Berechnen der ersten ableitung und sommit die geschwindigkeit, ausführung numerisch
    dv = [sd_num(step, sxv, syv, 0)]                                # geschwindigkeit variabel
    dc = [sd_num(step, sxc, syc, 0)]                                # geschwindigkeit konstan

    for i in range(1, len(step) - 1):
        dv.append(sd_num(step, sxv, syv, i))
        dc.append(sd_num(step, sxc, syc, i))

    dv.append(dv[-1])                                               # durch die differenatation verliere ich ein datenpunkt, daher wiederhole ich den letzen
    dc.append(dc[-1])

    # convert to array for further procesing
    dv_a = np.array(dv)*100
    dc_a = np.array(dc) * 100

    av = [sd_num(step, dv_a[:, 0], dv_a[:, 1], 0)]                   # die geschwindigkeit nochmals differenzieren um die beschleundigung zu erhalten
    ac = [sd_num(step, dc_a[:, 0], dc_a[:, 1], 0)]                   # die geschwindigkeit nochmals differenzieren um die beschleundigung zu erhalten

    for i in range(1, len(step) - 1):
        av.append(sd_num(step, dv_a[:, 0], dv_a[:, 1], i))          # Beschleunigung berechnen
        ac.append(sd_num(step, dc_a[:, 0], dc_a[:, 1], i))          # Beschleunigung berechnen
    av.append(av[-1])                                               # durch die differenatation verliere ich ein datenpunkt, daher wiederhole ich den letzen
    ac.append(av[-1])                                               # durch die differenatation verliere ich ein datenpunkt, daher wiederhole ich den letzen

    del dv_a, dc_a          #arrays löschen weil ich mit den listen weiter arbeite

    # settup des Plotes
    fig2, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim([min(x) - 50, max(x) + 50])
    ax.set_ylim([min(y) - 50, max(y) + 50])
    # Plotten der Punkte

    # Initialize the points
    point1, = ax.plot([1], [0], 'o', color='red')                       # 1 Punkt animation definieren
    point2, = ax.plot([1], [0], 'o', color='blue')                      # 2 Punkt animation definieren

    veloc_1 = ax.quiver([1], [0], [0], [-1], color='red', zorder=2)     # Geschwindigkeitvektoren definieren
    veloc_2 = ax.quiver([1], [0], [0], [-1], color='blue', zorder=2)

    acel_1 = ax.quiver([1], [0], [0], [-1], color='green', zorder=2)    # beschleunigumgsvektor definieren
    acel_2 = ax.quiver([1], [0], [0], [-1], color='orange', zorder=2)   # beschleunigumgsvektor definieren

    ax.plot(x, y, 'ro', label='Rohdaten')  # Gegebene punkt darstellen
    ax.plot(sxv, syv, zorder=1)  # Berechnete spline darstellen

    # Update-Funktion für die Animation
    def update(i):
        # Punkte aktualisieren
        point1.set_data(sxv[i + 1], syv[i + 1])
        point2.set_data(sxc[i + 1], syc[i + 1])

        # Pfeile aktualisieren
        veloc_1.set_offsets([sxv[i + 1], syv[i + 1]])
        veloc_2.set_offsets([sxc[i + 1], syc[i + 1]])
        veloc_1.set_UVC(dv[i][0], dv[i][1])  # dv[i] hat die form [u,v]
        veloc_2.set_UVC(dc[i][0], dc[i][1])  # dc[i] hat die form [u,v]

        acel_1.set_offsets([sxv[i + 1], syv[i + 1]])
        acel_1.set_UVC(av[i][0], av[i][1])  # dc[i] hat die form [u,v]

        acel_2.set_offsets([sxc[i + 1], syc[i + 1]])
        acel_2.set_UVC(ac[i][0], ac[i][1])  # dc[i] hat die form [u,v]

        # Rückgabe des geänderten Punkte-Objekts
        return point1, point2

    # Erstellung der Animation
    animation_1 = FuncAnimation(fig2, update, frames=len(step) - 1, interval=1, repeat=False)

    # Anzeigen der Animation
    plt.show()
    # setup movie save
    #animation_1.save("Projekt.gif")







def plot_Phi(phi, step):
    x = np.linspace(0, len(phi), len(step)) / 30
    fig1, ax1 = plt.subplots()
    ax1.set_xlim([min(x) - 5, max(x) + 5])
    ax1.set_ylim([min(phi) - 10, max(phi) + 10])
    ax1.plot(x, phi[:-3], 'ro', label='Phi')
    ax1.plot(x[:], step[:])
    plt.grid = True
    plt.show()


def sd_num(time, sx, sy, i):
    """
    :param time: Zeitvektor in diesem fall step weil wir diese zeit als referenz nehmen
    :param sx:   x-Koordinaten der spline (korrespondierend mit dem Zeitvektor)
    :param sy:   y-Koordinaten der spline (korrespondierend mit dem Zeitvektor)
    :param i:    index für den zeitvektor, an dem die ableitung berechnet werden sollte
    :return:     Ableitung an Phi[i]
    """

    # Catch division by zero
    # da wir die vektorn step un phi gleich lange machen mussten hat es am schluss 2-3 identische einträge daher kann es zu einer division durch 0 führen. das wird hier abgefangen
    if 0 == time[i] - time[i - 1]:
        return [0, 0]
    if 0 == time[i + 1] - time[i]:
        return [0, 0]

    if 0 < i < len(sx) - 1:
        # Ableitnug in x richtung
        dxn = (sx[i] - sx[i - 1]) / (time[i] - time[i - 1])  # Ableitung ein schritt nach hinten
        dxp = (sx[i + 1] - sx[i]) / (time[i + 1] - time[i])  # Ableitung ein schritt nach forne
        dx = (dxn + dxp) / 2  # Mittel

        # Ableitnug in y richtung
        dyn = (sy[i] - sy[i - 1]) / (time[i] - time[i - 1])  # Ableitung ein schritt nach hinten
        dyp = (sy[i + 1] - sy[i]) / (time[i + 1] - time[i])  # Ableitung ein schritt nach forne
        dy = (dyn + dyp) / 2  # Mittel
        return [dx / 100, dy / 100]
    if i == 0:  # Am ende wird der schritt nach hinten weggelassen
        dxp = (sx[i + 1] - sx[i]) / (time[i + 1] - time[i])  # Ableitung ein schritt nach forne
        dyp = (sy[i + 1] - sy[i]) / (time[i + 1] - time[i])  # Ableitung ein schritt nach forne
        return [dxp / 100, dyp / 100]
    if i == len(sx) - 1:  # Am ende wird der schritt nach vorne weggelassen
        dyn = (sy[i] - sy[i - 1]) / (time[i] - time[i - 1])  # Ableitung ein schritt nach hinten
        dxn = (sx[i] - sx[i - 1]) / (time[i] - time[i - 1])  # Ableitung ein schritt nach hinten
        return [dxn / 100, dyn / 100]
    else:
        return [0, 0]
