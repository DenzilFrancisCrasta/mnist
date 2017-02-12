import cPickle, gzip
import numpy as np

class MnistDataLoader(object):
    ''' MNIST Data Loader. 
        Preps the mnist digit data from a pickled file.  '''

    def __init__(self, mnist_path):
        self.mnist_path = mnist_path

    def dense_to_one_hot(self, labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def encode_one_hot(self, n):
        a = np.zeros((10, 1))
        a[n] = 1.0
        return a

    def load_and_prepare_data(self):
        ''' Load the mnist data from pickled file and 
            prepare dataset as list of tuples of predictor and target variables '''
        #Load the MNIST dataset
        f = gzip.open(self.mnist_path, 'rb')
        train_set, validation_set, test_set = cPickle.load(f)
        f.close()

        train_x = [np.reshape(x, (784, 1)) for x in train_set[0]]
        train_y = [self.encode_one_hot(y) for y in train_set[1]]
        train_data = zip(train_x, train_y)
        validation_x = [np.reshape(x, (784, 1)) for x in validation_set[0]]
        validation_data = zip(validation_x, validation_set[1]) 
        test_x = [np.reshape(x, (784, 1)) for x in test_set[0]]
        test_data = zip(test_x, test_set[1]) 
        return (train_data, validation_data, test_data)


