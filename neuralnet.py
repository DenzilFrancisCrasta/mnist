import numpy as np

class NeuralNetwork(object):
    ''' Multilayer Feedforward Neural Network trained using Stochastic Gradient Descent '''
    def __init__(self, sizes):
        # Set the random number seed for reproducibility of results
        np.random.seed(1234)

        self.sizes   = sizes
        self.biases  = [np.random.randn(x,1) for x   in sizes[1:] ]
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]

    def initialize_adam_parameters(self):
        self.prev_m_b = [np.zeros(b.shape) for b in self.biases]
        self.prev_v_b = [np.zeros(b.shape) for b in self.biases]
        self.prev_m_w = [np.zeros(w.shape) for w in self.weights]
        self.prev_v_w = [np.zeros(w.shape) for w in self.weights]

    def stochastic_gradient_descent(self, training_data, test_data, mini_batch_size, epochs, eta, gamma, nesterov=False, adam=False):
        ''' mini batch Stochastic Gradient Descent algorithm training ''' 

        for i in xrange(epochs):
            np.random.shuffle(training_data)
            self.prev_update_b = [np.zeros(b.shape) for b in self.biases]
            self.prev_update_w = [np.zeros(w.shape) for w in self.weights]

            if adam == True:
                self.initialize_adam_parameters()

            for j in xrange(0, len(training_data), mini_batch_size):
                self.process_mini_batch( training_data[j:j+mini_batch_size], eta, gamma, nesterov, adam)
            print "Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), len(test_data))


    def process_mini_batch(self, mini_batch, eta, gamma, nesterov, adam):
        ''' Process a single step of gradient descent on a mini batch '''
        epsilon = 1e-8 
        beta = (0.9, 0.999)

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        update_b = [np.zeros(b.shape) for b in self.biases]
        update_w = [np.zeros(w.shape) for w in self.weights]

        if adam == True:
            update_m_b = [np.zeros(b.shape) for b in self.biases]
            update_v_b = [np.zeros(b.shape) for b in self.biases]
            update_m_w = [np.zeros(w.shape) for w in self.weights]
            update_v_w = [np.zeros(w.shape) for w in self.weights]

        # Nesterov Lookahead Update 
        if nesterov == True:
            self.biases  = [b - gamma*prev_ub for b,prev_ub in zip(self.biases, self.prev_update_b)]
            self.weights = [w - gamma*prev_uw for w,prev_uw in zip(self.weights, self.prev_update_w)]

        # for each training data point P=(x,y) accumulate the derivative of error 
        for (x,y) in mini_batch:
			nabla_b_p, nabla_w_p = self.backpropogate(x,y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, nabla_b_p)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, nabla_w_p)]

        
        if adam == True:
            update_m_b = [ beta[0] * prev_mb + (1 - beta[0]) * nb for prev_mb,nb in zip(self.prev_m_b, nabla_b)]
            update_v_b = [ beta[1] * prev_vb + (1 - beta[1]) * (nb**2) for prev_vb,nb in zip(self.prev_v_b, nabla_b)]
            update_m_w = [ beta[0] * prev_mw + (1 - beta[0]) * nw for prev_mw,nw in zip(self.prev_m_w, nabla_w)]
            update_v_w = [ beta[1] * prev_vw + (1 - beta[1]) * (nw**2) for prev_vw,nw in zip(self.prev_v_w, nabla_w)]

        # calculate updates for the current mini batch, for non momentum methods gamma = 0
        if adam == True:
            update_b = [(eta / (len(mini_batch) * np.sqrt(u_vb + epsilon))) * u_mb for u_vb,u_mb in zip(update_v_b, update_m_b)]
            update_w = [(eta / (len(mini_batch) * np.sqrt(u_vw + epsilon))) * u_mw for u_vw,u_mw in zip(update_v_w, update_m_w)]
        else:
            update_b = [(int(not nesterov) * gamma * prev_b) + (eta/len(mini_batch))*nb for prev_b,nb in zip(self.prev_update_b, nabla_b)]
            update_w = [(int(not nesterov) * gamma * prev_w) + (eta/len(mini_batch))*nw for prev_w,nw in zip(self.prev_update_w, nabla_w)]

        self.biases = [b - ub for b,ub in zip(self.biases, update_b)]
        self.weights = [w - uw for w,uw in zip(self.weights, update_w)]

        if adam == True:
            self.prev_m_b = update_m_b
            self.prev_v_b = update_v_b 
            self.prev_m_w = update_m_w 
            self.prev_v_w = update_v_w 
        else:  
            self.prev_update_b = update_b
            self.prev_update_w = update_w

    def backpropogate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
  
        activation = x
        activations = [x]
        zs = []

        # forward propogate the input and store the activations and pre-activations lists 
        for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

        # Error at each output layer neuron is represented by delta 
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # Derivatives of the cost w.r.t weight and bias at the output layer 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Derivatives for all other layers below the output layer is calculated by backpropogation
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
        return sum([int(x==y) for (x,y) in results])


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))
