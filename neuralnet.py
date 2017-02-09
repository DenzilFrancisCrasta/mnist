import numpy as np

class NeuralNetwork(object):
    ''' Multilayer Feedforward Neural Network trained using Stochastic Gradient Descent '''
    def __init__(self, sizes):
        self.sizes   = sizes
        self.biases  = [np.random.randn(x,1) for x   in sizes[1:] ]
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]

    def stochastic_gradient_descent(self, training_data, test_data, mini_batch_size, epochs, eta):
        ''' mini batch Stochastic Gradient Descent algorithm training ''' 
        # Set the random number seed for reproducibility of results
        np.random.seed(1234)

        for i in xrange(epochs):
            np.random.shuffle(training_data)
            for j in xrange(0, len(training_data), mini_batch_size):
                self.process_step( training_data[j:j+mini_batch_size], eta )
            print "Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), len(test_data))


    def process_step(self, mini_batch, eta):
		''' Process a single step of gradient descent on a mini batch '''
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

        # for each training data point P=(x,y) accumulate the derivative of error 
		for (x,y) in mini_batch:
			nabla_b_p, nabla_w_p = self.backpropogate(x,y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, nabla_b_p)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, nabla_w_p)]

		self.weights = [w -(eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
		self.biases = [b -(eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]

    def backpropogate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
  
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, len(self.sizes)):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
		return (output_activations-y)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]  
        return sum(int(x==y) for (x,y) in results)


def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))
