''' MNIST Digit Classification Application '''
from mnist import MnistApplication
import cPickle, gzip

# Bootstrap the application
app = MnistApplication()
app.setup_commandline_parser()
app.parse_hyperparameters()

print(app.h_params)

#Load the MNIST dataset
f = gzip.open(app.h_params.mnist, 'rb')
train_set, validation_set, test_set = cPickle.load(f)
f.close()

#unpack the dataset predictors and target 
train_x, train_y = train_set
validation_x, validation_y = validation_set
test_x, test_y = test_set

print(len(train_x), len(train_y))
