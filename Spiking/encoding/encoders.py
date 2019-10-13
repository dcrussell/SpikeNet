import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp




def poisson(data, time, dt):
    """ Generates a poisson distributed spike trains for pixel data.
    The firing rate is based on the pixels intensity.

    Ref -  Fatahi et al 2016, evt_MNIST: A spike based version of traditional MNIST

    Input:
        data - The image to encode. Pixels should be non negative and in the range of (0, 1).
        time - The length of the spike train (in ms).
        dt   - The Inter-spike interval (small time frame which we wish to check whether a spike will be emitted or not) (i.e. 1 ms)

    Return:
        spikes - Shape of (data_dim0, data_dim1, num
    """
    shape = tf.shape(data).numpy()
    size = tf.size(data).numpy()
    data = tf.reshape(data, [-1]).numpy()

    num_bins = int(time/dt)

    randnums = np.random.rand(size, num_bins)

    spikes = np.zeros((size, num_bins))



    for i in range(size):
        rate = data[i] * dt
        for j in range(num_bins):
            if rate > randnums[i][j]:
                spikes[i][j] = 1



    spikes = np.reshape(spikes, (shape[0], shape[1], num_bins))

    return spikes






if __name__ == '__main__':

    spikes = poisson_encoder(tf.constant([.5,.2,0,.6,.9,.3], shape=(2,3)), 50, 1.0)
