''' MNIST Digit Classification Application '''
from parser import CommandLineParser
from mnist import MnistDataLoader
from neuralnet import NeuralNetwork

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

if nesterov:
    print "Nesterov Gradient Descent Mode Activated "

neural_net = NeuralNetwork(parser.h_params.sizes)
neural_net.stochastic_gradient_descent(training, validation, parser.h_params.batch_size, 30, parser.h_params.lr, parser.h_params.momentum, nesterov) 
