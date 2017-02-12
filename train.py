''' MNIST Digit Classification Application '''
from parser import CommandLineParser
from mnist import MnistDataLoader
from neuralnet import NeuralNetwork
import activations as act_f
import costs

import numpy as np


# Setup the parser to read hyperparams from commandline 
parser = CommandLineParser()
parser.initialize_switches()
parser.parse_hyperparameters()

print(parser.h_params)

# Load the mnist dataset
loader = MnistDataLoader(parser.h_params.mnist)
training, validation, testing = loader.load_and_prepare_data()

nesterov = True if parser.h_params.opt == "nag" else False
adam = True if parser.h_params.opt == "adam" else False

if nesterov:
    print "Nesterov Gradient Descent Mode Activated "
elif adam:
    print "Adam based Gradient Descent Mode Activated"

if parser.h_params.activation == 'sigmoid':
    activation_function, activation_prime = act_f.sigmoid, act_f.sigmoid_prime
else:
    activation_function, activation_prime = act_f.tanh, act_f.tanh_prime
    
if parser.h_params.loss == 'sq':
    loss = costs.SquaredError(activation_prime)
else:
    loss = costs.CrossEntropy(activation_prime)


neural_net = NeuralNetwork(parser.h_params.sizes, loss, activation_function, activation_prime)
neural_net.stochastic_gradient_descent(training, validation, parser.h_params.batch_size, 1000, parser.h_params.lr, parser.h_params.momentum, 0.5, nesterov, adam) 
