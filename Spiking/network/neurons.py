import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt



#TODO:

class LIFNeuron:
    pass


class SRMNeuron:
    """Spike Response Model with Escape Noise.

    Ref: Jang, Simeone, Gardner, and Gruning -
            'An Introduction to Spiking Neural Networks: Probabilist Models,
            Learning Rules and Applications'


    """

    def __init__(self,
                 n_connections,
                 name,
                 rest = -70,
                 dt = 0.1,
                 feedforward_td = 0.1,
                 feedback_td = 0.1,
                 window = 100
                 ):

        # membrane potential
        self.rest = rest
        self.name = name
        self.window_size = window
        self.potential = rest


        # weights for connections to presynaptic neurons
        self.ex_weights = np.ones((n_connections, 1)) #np.random.randn(n_connections, 1) * np.sqrt(2/(1+n_connections))
        self.int_weight = 1 # np.random.rand()
        self.gamma = 0
        # all spikes emitted by neuron up to time t
        # start with 0 since there would be no activity before input is
        # recieved
        self.spikes = [0]

        # simulation step size
        self.dt = dt

        # feed forward and feedback trace decay
        self.ff_td = feedforward_td
        self.fb_td = feedback_td


    def _a(self, t):
        # exponential kernel for feedforward spikes
        return (np.exp(-t) - np.exp(-t/self.ff_td))

    def _b(self, t):

        # exponential kernel for feedback connection
        return (-np.exp(-t))

    def _sigmoid(self, x):

        return 1 / (1+ np.exp(-x/4))

    def forward(self, x, return_filters=True):

        # get windowed view for spikes
        x_shape = x.shape
        x = x[:, max(0, x_shape[1]-self.window_size): max(0, x_shape[1]-self.window_size) + self.window_size]
        potential = self.rest + self.gamma
        for i in range(x.shape[0]):

            # x is indexed at -1-j since the most recent spikes will be at the end
            # probably a better way to go about this such as having the network
            # reverse each nodes spike train before feeding it to the next node
            # vectorized??
            filtered_forward = 0
            filtered_feedback = 0
            for j in range(x.shape[1]):
                filtered_forward += self._a((j)*self.dt)* x[i, -1-j]
            for j in range(len(self.spikes)):
                filtered_feedback += self._b((j) * self.dt) * self.spikes[-1-j]

            potential += (self.ex_weights[i] * filtered_forward + self.int_weight * filtered_feedback)

        sig = self._sigmoid(potential)

        dist = tfp.distributions.Bernoulli(probs=sig)

        spike = dist.sample().numpy()

        self.spikes.append(int(spike))

        return int(spike), potential
