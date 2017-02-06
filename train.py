import cPickle, gzip
import numpy as np
import argparse

#Load the MNIST dataset
#f = gzip.open('mnist.pkl.gz', 'rb')
#train_set, valid_set, test_set = cPickle.load(f)
#f.close()

#data, label = train_set
#print(len(data), len(label))

def valid_batch_size(string):
	''' Enforces valid mini-batch-sizes for gradient descent algorithms '''
	batch_size = int(string)
	if batch_size != 1 and batch_size % 5 != 0:
		msg = "valid batch size is 1 or multiples of 5. %r does not satisfy these constraints" % string
		raise argparse.ArgumentTypeError(msg)
	return batch_size

def bool_t(string):
	''' Converts a string with true or false values into bool '''
	return True if string == 'true' else False

def csv_to_list(string):
	''' Converts a csv format string holding ints into a python list ''' 
	return [int(x) for x in string.split(',')]

#Initialize switches 

help_msgs = {
	'learning_rate': 'initial learning rate for gradient descent based algorithms',
	'momentum'     : 'momentum to be used by momentum based algorithms',
	'num_hidden'   : 'number of hidden layers not including the input layer and output layer',
	'sizes'        : 'a comma separated list for the size of each hidden layer',
	'activation'   : 'choice of the activation function ',
	'loss'         : 'possible choices are squared error[sq] or cross entropy[ce]',
    'opt'          : 'optimization algorithm to be used',
	'batch_size'   : 'batch size to be used',
	'anneal'       : 'choosing anneal will halve the learning rate when validation loss increases in any epoch and then restart that epoch',
	'save_dir'     : 'directory in which the model will be saved',
	'export_dir'   : 'directory in which the log files will be saved',
	'mnist'        : 'path to the mnist data in pickeled format'
}

parser = argparse.ArgumentParser()

parser.add_argument('--lr'        , type = float, default=2.5, help = help_msgs['learning_rate'])
parser.add_argument('--momentum'  , type = float, help = help_msgs['momentum'])
parser.add_argument('--num_hidden', type = int  , default=1, help = help_msgs['num_hidden'])
parser.add_argument('--sizes'     , type = csv_to_list, help = help_msgs['sizes'])
parser.add_argument('--activation', choices = ['tanh', 'sigmoid'], default = 'sigmoid', help = help_msgs['activation'])
parser.add_argument('--loss'      , choices = ['sq', 'ce'], default = 'sq', help = help_msgs['loss'])
parser.add_argument('--opt'       , choices = ['gd', 'momentum', 'nag', 'adam'], default = 'gd', help = help_msgs['opt'])
parser.add_argument('--batch_size', type = valid_batch_size, help = help_msgs['batch_size'])
parser.add_argument('--anneal'    , type = bool_t, default=False, help = help_msgs['anneal'])
parser.add_argument('--save_dir'  , help = help_msgs['save_dir'], default='.')
parser.add_argument('--expt_dir'  , help = help_msgs['export_dir'], default='.')
parser.add_argument('--mnist'     , help = help_msgs['mnist'], required=True)

args = parser.parse_args()
print(args)
