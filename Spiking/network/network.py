import tensorflow as tf
import numpy as np

class Input:
    def __init__(self, x):
        self.data = x
        self.input_shape = x.shape
        self.duration = self.input_shape[1]
#TODO: proto
class Network:
    def __init__(self):
        self.time = 0
        self.connections = {}
        self.spikes = {}

    def add_connection(self, pre, post):
        # pre -> post connection
        self.connections[pre] = post


    def forward(self):
        pass

# prototype connection for single neurons
class Connection:

    def __init__(self, neuron_a, neuron_b):
        pass
