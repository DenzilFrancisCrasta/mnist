import argparse, neuralnet
    
class CommandLineParser(object):
    ''' Command Line Parser to parse the variety of arguments furnished to
        train a MultiLayer Feedforward Neural Network using Stochastic Gradient Descent 
        for classification of MNIST digit dataset ''' 

    # help messages for the various input parameters 

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
        'mnist'        : 'path to the mnist data in pickeled format',
        'lambda'       : 'regularization parameter',
        'epochs'       : 'max epochs'
    }

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize_switches(self):
        ''' Setup the parser to accept hyperparameters from commandline arguments '''

        # helper functions to check constraints and format certain command-line arguments
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

        self.parser.add_argument('--lr'        , type = float, default=2.5, help = self.help_msgs['learning_rate'])
        self.parser.add_argument('--momentum'  , type = float, default=0, help = self.help_msgs['momentum'])
        self.parser.add_argument('--num_hidden', type = int  , default=1, help = self.help_msgs['num_hidden'])
        self.parser.add_argument('--sizes'     , type = csv_to_list, help = self.help_msgs['sizes'])
        self.parser.add_argument('--activation', choices = ['tanh', 'sigmoid'], default = 'sigmoid', help = self.help_msgs['activation'])
        self.parser.add_argument('--loss'      , choices = ['sq', 'ce'], default = 'sq', help = self.help_msgs['loss'])
        self.parser.add_argument('--opt'       , choices = ['gd', 'momentum', 'nag', 'adam'], default = 'gd', help = self.help_msgs['opt'], required=True)
        self.parser.add_argument('--batch_size', type = valid_batch_size, default = 10, help = self.help_msgs['batch_size'])
        self.parser.add_argument('--anneal'    , type = bool_t, default=False, help = self.help_msgs['anneal'])
        self.parser.add_argument('--save_dir'  , help = self.help_msgs['save_dir'], default='.')
        self.parser.add_argument('--expt_dir'  , help = self.help_msgs['export_dir'], default='.')
        self.parser.add_argument('--mnist'     , help = self.help_msgs['mnist'], required=True)
        self.parser.add_argument('--lmbda'     , type = float, default=0.5, help = self.help_msgs['lambda'])
        self.parser.add_argument('--epochs'    , type = int, default=30, help = self.help_msgs['epochs'])

    def are_params_valid(self):
        msg = None

        if len(self.h_params.sizes) != self.h_params.num_hidden:
            msg = "Mismatch between number of hidden layers and list length of hidden layer sizes" 
        
        if self.h_params.opt in ['momentum','nag'] and self.h_params.momentum == 0:
            msg = "Plese supply a momentum using --momentum switch for the {} optimization method chosen ".format(self.h_params.opt) 

        if self.h_params.opt not in ['momentum','nag'] and self.h_params.momentum > 0:
            msg = "Please choose a momentum based optimization method. {} does not make use of the momentum {} supplied".format(self.h_params.opt, self.h_params.momentum)  

        if msg is not None:
            self.parser.error(msg)

    def parse_hyperparameters(self):
        ''' Parse the hyperparameters for training using the command line arguments '''
        self.h_params = self.parser.parse_args()
        self.are_params_valid()
        # add the sizes of input and output layer to the sizes list
        self.h_params.sizes.insert(0, 784)
        self.h_params.sizes.append(10)


if __name__ == '__main__':
    app = CommandLineParser()
    app.initialize_switches()
    app.parse_hyperparameters()
    print(app.h_params)


