''' MNIST Digit Classification Application '''
from parser import CommandLineParser
from mnist import MnistApplication
import numpy as np

# Setup the parser to read hyperparams from commandline 
parser = CommandLineParser()
parser.initialize_switches()
parser.parse_hyperparameters()

# add the sizes of input and output layer to the sizes list
parser.h_params.sizes.insert(0, 784)
parser.h_params.sizes.append(10)

# Bootstrap the mnist application
app = MnistApplication(parser.h_params)
app.load_and_prepare_data()
app.train()

print(parser.h_params)

