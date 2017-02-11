''' Cost Functions Module '''

import numpy as np

class CostFunction(object):
    def __init__(self, activation_prime):
        pass

    def cost(self, target, output, weighted_input):
        pass

    def delta(self, target, output, weighted_input):
        pass


class CrossEntropy(CostFunction):
    def __init__(self, activation_prime):
        self.activation_prime = activation_prime

    def cost(self, target, output, weighted_input):
        pass

    def delta(self, target, output, weighted_input):
        return output - target 

class SquaredError(CostFunction):
    def __init__(self, activation_prime):
        self.activation_prime = activation_prime

    def cost(self, target, output, weighted_input):
        pass

    def delta(self, target, output, weighted_input):
        return (output - target) * self.activation_prime(weighted_input) 
