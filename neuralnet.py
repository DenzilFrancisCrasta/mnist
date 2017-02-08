import numpy as np

class NeuralNetwork(object):
    ''' Multilayer Feedforward Neural Network trained using Stochastic Gradient Descent '''
    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]
        np.random.seed(1234)

    def stochastic_gradient_descent(self, training_data, epochs, eta):
        for i in xrange(epochs):
            np.random.shuffle(training_data)



