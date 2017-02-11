''' Activation Functions '''
import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
    return (np.exp(2*z) - 1) /(np.exp(2*z) + 1)

def tanh_prime(z):
    return 1 - tanh(z)**2
