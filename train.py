''' MNIST Digit Classification Application '''
from parser import CommandLineParser
from mnist import MnistDataLoader
from neuralnet import NeuralNetwork
import activations as act_f
import logger as lg
import costs

import numpy as np


# Setup the parser to read hyperparams from commandline 
parser = CommandLineParser()
parser.initialize_switches()
parser.parse_hyperparameters()

#print(parser.h_params)

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
    loss = costs.SoftmaxSquaredError(activation_prime)
else:
    loss = costs.CrossEntropy(activation_prime)

#setup loggers 
loss_formatter  = lg.Formatter("Epoch {}, Step {}, Loss: {:.2f}, lr: {}\n")
error_formatter = lg.Formatter("Epoch {}, Step {}, Error: {:.2f}, lr: {}\n")

build_logs = True
if build_logs == True:
    loggers = {
                'train_loss_logger'  : lg.Logger(parser.h_params.expt_dir + '/log_loss_train.txt', loss_formatter),
                'valid_loss_logger'  : lg.Logger(parser.h_params.expt_dir + '/log_loss_valid.txt', loss_formatter),  
                'test_loss_logger'   : lg.Logger(parser.h_params.expt_dir + '/log_loss_test.txt' , loss_formatter),  
                'train_error_logger' : lg.Logger(parser.h_params.expt_dir + '/log_error_train.txt', error_formatter),
                'valid_error_logger' : lg.Logger(parser.h_params.expt_dir + '/log_error_valid.txt', error_formatter),  
                'test_error_logger'  : lg.Logger(parser.h_params.expt_dir + '/log_error_test.txt' , error_formatter)  
              }
else:
    loggers = {}

neural_net = NeuralNetwork(parser.h_params.sizes, 
                           loss, 
                           activation_function, 
                           activation_prime, 
                           act_f.softmax, 
                           loggers,
                           parser.h_params.expt_dir, parser.h_params.save_dir,
                           parser.h_params.anneal)

neural_net.stochastic_gradient_descent(training, validation, testing, 
                                       parser.h_params.batch_size, parser.h_params.epochs, 
                                       parser.h_params.lr, parser.h_params.momentum, parser.h_params.lmbda, 
                                       nesterov, adam,
                                       build_logs) 
