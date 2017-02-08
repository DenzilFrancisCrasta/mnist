import cPickle, gzip

class MnistApplication(object):
    ''' MNIST Application Driver. 
        Responsibilities include setting up the hyperparameters and 
        preparing datasets before delegating the training to 
        the MultiLayer Feedforward Neural Network '''

    def __init__(self, hyper_params):
        self.h_params = hyper_params

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

        # pack data into a list of predictor and target tuple list
        self.training_data = [(x, y) for x, y in zip(train_x, train_y)]
        self.validation_data = [(x, y) for x, y in zip(validation_x, validation_y)]
        self.test_data = [(x, y) for x, y in zip(test_x, test_y)]
        print(len(self.training_data))

#np.random.seed(1234)
#np.random.shuffle(training_data)
#print(training_data[0], len(training_data))



