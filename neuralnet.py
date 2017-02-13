import numpy as np
import cPickle

class NeuralNetwork(object):
    ''' Multilayer Feedforward Neural Network trained using Stochastic Gradient Descent '''
    def __init__(self, sizes, cost, activation_function, activation_prime, output_activation_function, loggers, expt_dir, save_dir, anneal=False):
        # Set the random number seed for reproducibility of results
        np.random.seed(1234)

        self.sizes   = sizes
        self.cost = cost
        self.expt_dir = expt_dir
        self.save_dir = save_dir
        self.anneal = anneal
        self.loggers = loggers
        self.activation_function = activation_function
        self.output_activation_function = output_activation_function
        self.activation_prime = activation_prime
        self.biases  = [np.random.randn(x,1)*(1.0/400) for x   in sizes[1:] ]
        self.weights = [np.random.randn(x,y)*(1.0/400) for x,y in zip(sizes[1:], sizes[:-1])]
        self.prev_b = [np.zeros(b.shape) for b in self.biases]
        self.prev_w = [np.zeros(w.shape) for w in self.weights]
        self.best_biases = [np.zeros(b.shape) for b in self.biases]
        self.best_weights = [np.zeros(w.shape) for w in self.weights]

    def initialize_adam_parameters(self):
        self.prev_m_b = [np.zeros(b.shape) for b in self.biases]
        self.prev_v_b = [np.zeros(b.shape) for b in self.biases]
        self.prev_m_w = [np.zeros(w.shape) for w in self.weights]
        self.prev_v_w = [np.zeros(w.shape) for w in self.weights]

    def stochastic_gradient_descent(self, training_data, validation_data, test_data, 
                                    mini_batch_size, epochs, 
                                    eta, gamma, lmbda, 
                                    nesterov=False, adam=False, build_logs=False):
        ''' mini batch Stochastic Gradient Descent algorithm training ''' 

        self.prev_validation_loss = 1000000
        self.best_val_accuracy = 0
        for i in xrange(epochs):
            np.random.shuffle(training_data)

            self.prev_update_b = [np.zeros(b.shape) for b in self.biases]
            self.prev_update_w = [np.zeros(w.shape) for w in self.weights]

            if adam == True:
                self.initialize_adam_parameters()

            step = 1
            for j in xrange(0, len(training_data), mini_batch_size):
                self.process_mini_batch( training_data[j:j+mini_batch_size], eta, gamma, lmbda, nesterov, adam, len(training_data))
                if build_logs == True and step % 100 == 0:
                    self.loggers['train_loss_logger'].log([i, step, self.total_cost(training_data[:10001]), eta])
                    self.loggers['valid_loss_logger'].log([i, step, self.total_cost(validation_data, True), eta])
                    self.loggers['test_loss_logger'].log([i, step, self.total_cost(test_data, True), eta])
                    self.loggers['train_error_logger'].log([i, step, self.error_rate(training_data[:10001], True), eta])
                    self.loggers['valid_error_logger'].log([i, step, self.error_rate(validation_data), eta])
                    self.loggers['test_error_logger'].log([i, step, self.error_rate(test_data), eta])
                step += 1

            if self.anneal == True:
                current_validation_loss = self.total_cost(validation_data, True)
                if current_validation_loss > self.prev_validation_loss: 
                    eta = max(eta * 0.5, 0.00005) 
                    for level in xrange(len(self.weights)):
                        np.copyto(self.weights[level], self.prev_w[level])
                        np.copyto(self.biases[level], self.prev_b[level])
                else:
                    for level in xrange(len(self.weights)):
                        np.copyto(self.prev_w[level], self.weights[level])
                        np.copyto(self.prev_b[level], self.biases[level])
                    self.prev_validation_loss = current_validation_loss

            self.current_val_accuracy = self.evaluate(validation_data)
            if self.current_val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = self.current_val_accuracy
                for level in xrange(len(self.weights)):
                    np.copyto(self.best_weights[level], self.weights[level])
                    np.copyto(self.best_biases[level], self.biases[level])

            print "Epoch {0}: {1} out of {2} correct with lr:{3}".format(i, self.current_val_accuracy, len(validation_data), eta)
        for level in xrange(len(self.weights)):
            np.copyto(self.weights[level], self.best_weights[level])
            np.copyto(self.biases[level], self.best_biases[level])
        self.save_predictions_and_pickle(validation_data, test_data)

    def process_mini_batch(self, mini_batch, eta, gamma, lmbda, nesterov, adam, n):
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
            update_b = [(eta / (len(mini_batch) * (np.sqrt(u_vb) + epsilon))) * u_mb for u_vb,u_mb in zip(update_v_b, update_m_b)]
            update_w = [(eta / (len(mini_batch) * (np.sqrt(u_vw) + epsilon))) * u_mw for u_vw,u_mw in zip(update_v_w, update_m_w)]
        else:
            update_b = [(int(not nesterov) * gamma * prev_b) + (eta/len(mini_batch))*nb for prev_b,nb in zip(self.prev_update_b, nabla_b)]
            update_w = [(int(not nesterov) * gamma * prev_w) + (eta/len(mini_batch))*nw for prev_w,nw in zip(self.prev_update_w, nabla_w)]

        self.biases = [b - ub for b,ub in zip(self.biases, update_b)]
        self.weights = [(1 -  eta * (lmbda/n))*w - uw for w,uw in zip(self.weights, update_w)]

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
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        zs.append(z)
        activation = self.output_activation_function(z)
        activations.append(activation)
        

        # Error at each output layer neuron is represented by delta 
        delta = self.cost.delta(y, activations[-1], zs[-1])
        # Derivatives of the cost w.r.t weight and bias at the output layer 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Derivatives for all other layers below the output layer are calculated by backpropogation
        for l in xrange(2, len(self.sizes)):
			z = zs[-l]
			sp = self.activation_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def encode_one_hot(self, n):
        a = np.zeros((10, 1))
        a[n] = 1.0
        return a
  
    def total_cost(self, data, make_one_hot=False):
        total_cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if make_one_hot: y = self.encode_one_hot(y)
            total_cost += self.cost.cost(y, a)
        return total_cost
    
    def feedforward(self, a):
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = self.activation_function(np.dot(w, a) + b)
        a = self.output_activation_function(np.dot(self.weights[-1],a)+self.biases[-1])
        return a

    def error_rate(self, data, is_one_hot=False):
        if is_one_hot == True:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x,y) in data]  
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x,y) in data]  
        correct = sum([int(x==y) for (x,y) in results])

        return 100 * (float(len(data) - correct)/ len(data)) 

    def predict_labels(self, data):
        results = [np.argmax(self.feedforward(x)) for (x,y) in data]  
        return results
    
    def save_predictions_and_pickle(self, validation_data, test_data):
        np.savetxt(self.expt_dir +'/valid_predictions.txt', self.predict_labels(validation_data), fmt='%d')
        np.savetxt(self.expt_dir+'/test_predictions.txt', self.predict_labels(test_data), fmt='%d')
        cPickle.dump((self.weights, self.biases), open(self.save_dir+'/model_weight_biases_tuple.pkl', 'wb'))

    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]  
        return sum([int(x==y) for (x,y) in results])
