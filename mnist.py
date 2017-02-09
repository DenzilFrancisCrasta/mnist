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

    def dense_to_one_hot(self, labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

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

        train_y = self.dense_to_one_hot(train_y, 10)
        validation_y = self.dense_to_one_hot(validation_y, 10)
        test_y = self.dense_to_one_hot(test_y, 10)

        # pack data into a list of predictor and target tuple list
        self.training_data = [(x, y) for x, y in zip(train_x, train_y)]
        self.validation_data = [(x, y) for x, y in zip(validation_x, validation_y)]
        self.test_data = [(x, y) for x, y in zip(test_x, test_y)]
        #print(self.training_data[0])

    def train(self):
		nn = NeuralNetwork(self.h_params.sizes)
		nn.stochastic_gradient_descent(self.training_data, self.validation_data, self.h_params.batch_size, 30, self.h_params.lr) 
