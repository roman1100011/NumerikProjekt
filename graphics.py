
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def plot_raw(t, x, y):
    # create the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 800])

    # create the line plot
    line, = ax.plot(x[:1], y[:1])

    # update function for the animation
    def update(frame):
        # update the data of the line plot
        line.set_data(x[:frame+1], y[:frame+1])
        # set the color of the line plot based on time
        line.set_color(plt.cm.RdYlBu(frame/len(t)))
        # set the linewidth of the line plot based on time
        line.set_linewidth(3*(frame/len(t)))
        return line,

    # create the animation
    animation = FuncAnimation(fig, update, frames=len(t), interval=200)

    # show the animation
    plt.show()


