import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from utils.plot import plot_spikes
from network.neurons import SRMNeuron


neuron = SRMNeuron(30, name="neuron_a")
# Simulate spike trains as input
dist = tfp.distributions.Bernoulli(probs=0.5)
input = []
for i in range(30):
    input.append(dist.sample(200))
input = np.array(input)

#collect feedforwards, feedbacks, spikes, and voltage potentials
feedforwards = []
feedbacks = []
potentials = []
inpt = None
sp = []

# run forward through time
for step in range(input.shape[1]):
    spike_step = np.expand_dims(input[:, step], axis=1)
    if step == 0:
        inpt = spike_step
    else:
        inpt = np.concatenate((inpt, spike_step), axis=1)
    spike, potential = neuron.forward(inpt)
    sp.append(spike)
    potentials.append(potential)

# plot
x = np.arange(0, len(potentials), 1)
plt.plot(x, potentials)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron voltage potential")
plt.show()

plt.plot(x, sp)
plt.xlabel("Time (ms)")
plt.ylabel("Spike")
plt.show()
