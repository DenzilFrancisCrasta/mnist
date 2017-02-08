''' MNIST Digit Classification Application '''
from parser import CommandLineParser
from mnist import MnistApplication
import numpy as np

# Setup the parser to read hyperparams from commandline 
parser = CommandLineParser()
parser.initialize_switches()
parser.parse_hyperparameters()

# Bootstrap the mnist application
app = MnistApplication(parser.h_params)
app.load_and_prepare_data()

print(parser.h_params)

