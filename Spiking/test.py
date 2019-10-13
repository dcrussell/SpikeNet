import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from utils.plot import plot_spikes
from network.nodes import SRMNeuron


neuron = SRMNeuron(80)


dist = tfp.distributions.Bernoulli(probs=0.2)
input = []
for i in range(50):
    input.append(dist.sample(200))

input = np.array(input)

ff = []
fb = []
pt = []
inpt = None
sp = []
for step in range(input.shape[1]):
    spike_step = np.expand_dims(input[:, step], axis=1)
    if step == 0:
        inpt = spike_step
    else:

        inpt = np.concatenate((inpt, spike_step), axis=1)
    spike, potential = neuron.forward(inpt)
    sp.append(spike)

    pt.append(potential)



x = np.arange(0, len(pt), 1)
plt.plot(x, pt)
plt.show()

plt.plot(x, sp)
plt.show()
