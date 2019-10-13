import matplotlib.pyplot as plt
import numpy as np



def plot_spikes(spikes):
    spikes = np.array(spikes)
    spikes = np.expand_dims(spikes, axis=0)
    print(spikes.shape)
    fig = plt.figure(dpi=120)
    axes = fig.add_subplot(111)


    for i in range(spikes.shape[1]):

        axes.plot(spikes[0, i], 0, 'b.')

    axes.set_xlabel("Seconds")
    axes.set_ylabel("Cell Number")
    plt.xticks([i for i in range(spikes.shape[1])])
    plt.draw()
    plt.show()
