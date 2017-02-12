''' Cost Functions Module '''

import numpy as np

class CostFunction(object):
    def __init__(self, activation_prime):
        pass

    def cost(self, target, output):
        pass

    def delta(self, target, output, weighted_input):
        pass


class CrossEntropy(CostFunction):
    def __init__(self, activation_prime):
        self.activation_prime = activation_prime

    def cost(self, target, output):
        return -np.sum(target*np.log(output))

    def delta(self, target, output, weighted_input):
        return output - target 

class SquaredError(CostFunction):
    def __init__(self, activation_prime):
        self.activation_prime = activation_prime

    def cost(self, target, output):
        return 0.5 * np.sum((target - output)**2)

    def delta(self, target, output, weighted_input):
        return (output - target) * self.activation_prime(weighted_input) 

class SoftmaxSquaredError(CostFunction):
    def __init__(self, activation_prime):
        self.activation_prime = activation_prime

    def cost(self, target, output):
        return np.sum((target - output)**2)

    def delta(self, target, output, weighted_input):
        return np.dot((np.identity(len(target)) - output) * (output - target).transpose(), output)
