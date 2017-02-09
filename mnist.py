import cPickle, gzip
from neuralnet import NeuralNetwork
import numpy as np

class MnistApplication(object):
    ''' MNIST Application Driver. 
        Responsibilities include setting up the hyperparameters and 
        preparing datasets before delegating the training to 
        the MultiLayer Feedforward Neural Network '''

    def __init__(self, hyper_params):
        self.h_params = hyper_params

    def make_components(self, data_xy):  
        x, y = data_xy
        nx = np.asarray(x)
        print(nx.shape)
        ny = np.asarray(y)
        return (nx, ny)

    def load_and_prepare_data(self):
        ''' Load the mnist data from pickled file and 
            prepare dataset as list of tuples of predictor and target variables '''

        #Load the MNIST dataset
        f = gzip.open(self.h_params.mnist, 'rb')
        train_set, validation_set, test_set = cPickle.load(f)
        f.close()

        #unpack the train, validation and test data tuples
        train_x, train_y = train_set
        validation_x, validation_y = validation_set
        test_x, test_y = test_set

        self.training_data = train_set
        self.validation_data = validation_set
        self.test_data = test_set

        # pack data into a list of predictor and target tuple list
        #self.training_data = [(np.asarray(x), y) for x, y in zip(train_x, train_y)]
        #self.validation_data = [(np.asarray(x), y) for x, y in zip(validation_x, validation_y)]
        #self.test_data = [(np.asarray(x), y) for x, y in zip(test_x, test_y)]
        #print(self.training_data[0])

    def train(self):
		nn = NeuralNetwork(self.h_params.sizes)
		nn.stochastic_gradient_descent(self.training_data, self.validation_data, self.h_params.batch_size, 30, self.h_params.lr) 
