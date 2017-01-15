"""Search initial values of parameters in our simple neural network"""

import math
import random
import numpy as np


class SimpleNeuralNetwork(object):
    """Simple neural network"""

    def __init__(self):
        self.lamb = np.random.randn() # pylint: disable = no-member
        self.bias = np.random.randn() # pylint: disable = no-member
        self.weight = np.random.randn() # pylint: disable = no-member

        self.lamb_init = self.lamb
        self.bias_init = self.bias
        self.weight_init = self.weight

        print "self.lamb = ", self.lamb
        print "self.bias = ", self.bias
        print "self.weight = ", self.weight


    def feedforward(self, coordinate_x, coordinate_y):
        """feedforward throuth the simple neural network"""

        return sigmoid(coordinate_y - self.lamb * math.pow((self.weight * coordinate_x + \
            self.bias), 2))

    def loss_fun(self, training_data):
        """Compute the loss function"""

        loss = 0.0
        for coordinate_x, coordinate_y, label in training_data:
            weighted_z = coordinate_y - self.lamb * math.pow((self.weight * coordinate_x + \
                self.bias), 2)
            z_sig = sigmoid(weighted_z)
            loss = loss - label * np.log(z_sig) - (1-label) * np.log(1 - z_sig) # pylint: disable = no-member
        return loss

    def gradient_descent(self, training_data, epochs, eta, test_data):
        """Gradient descent on the simple neural network"""

        num_test = len(test_data)
        for _ in xrange(epochs):
            random.shuffle(training_data)
            self.update_params(training_data, eta)
        temp_test_accu = 1.0 * self.evaluate(test_data)/num_test
        return temp_test_accu, self.lamb_init, self.bias_init, self.weight_init


    def update_params(self, training_data, eta):
        """Update parameters"""

        nabla_lamb = 0.0
        nabla_bias = 0.0
        nabla_weight = 0.0
        for coordinate_x, coordinate_y, label in training_data:
            delta_nabla_lamb, delta_nabla_bias, delta_nabla_weight = self.backprop(coordinate_x, \
                coordinate_y, label)
            nabla_lamb = nabla_lamb + delta_nabla_lamb
            nabla_bias = nabla_bias + delta_nabla_bias
            nabla_weight = nabla_weight + delta_nabla_weight
        self.lamb = self.lamb - (eta/len(training_data)) * nabla_lamb
        self.bias = self.bias - (eta/len(training_data)) * nabla_bias
        self.weight = self.weight - (eta/len(training_data)) * nabla_weight


    def backprop(self, coordinate_x, coordinate_y, label):
        """Back propagation on the simple neural network"""

        # feedforward
        temp_u1 = coordinate_x * self.weight
        temp_u2 = temp_u1 + self.bias
        temp_u3 = math.pow(temp_u2, 2)
        temp_u4 = temp_u3 * self.lamb
        weighted_z = coordinate_y - temp_u4

        # backward pass
        temp = (label - sigmoid(weighted_z))
        nabla_lamb = temp * temp_u1
        nabla_bias = temp * self.lamb * 2 * temp_u2
        nabla_weight = nabla_bias * coordinate_x

        return (nabla_lamb, nabla_bias, nabla_weight)

    def evaluate(self, test_data):
        """Evaluate the neural network"""

        num_correct_predict = 0
        test_results = [(self.feedforward(x, y), label) for (x, y, label) in test_data]
        for predict_prob, label in test_results:
            if abs(label - predict_prob) < 0.5:
                num_correct_predict += 1
        return num_correct_predict


def sigmoid(weighted_z):
    """Compute sigmoid function"""

    return 1.0/(1.0+np.exp(- weighted_z)) # pylint: disable = no-member


if __name__ == '__main__':
    # lamb_best = 0.0
    # bias_best = 0.0
    # weight_best = 0.0
    TEST_ACCU_BEST = 0.0
    ITERATIONS = 100000
    LEARNING_RATE = 0.001
    NUM_SEARCH = 20000000000

    # Training data: in each element, [coordinate x, coordinate y, label]
    TRAINING_DATA = np.array([[-0.5, 12, 0], [0.5, 13.2, 1], [0.8, 8, 0], [1, 9, 1], \
        [1.3, 6.5, 1], [1.5, 5, 1], [1.7, 3, 0], [2.0, 1.5, 0], [2.2, 2, 1], [2.5, 1, 1], \
        [3, -1.3, 0], [3.3, 0.5, 1], [3.5, 1.5, 1], [3.8, 1.1, 0], [4.1, 2.8, 1], [4.5, 5.1, 1], \
        [4.9, 6.8, 0], [5.3, 11.2, 1], [5.7, 14.1, 0], [6.2, 21.2, 1]])

    # Test data
    TEST_DATA = np.array([[0.7, 11.1, 1], [1.3, 5.2, 0], [1.7, 3.0, 0], [2.2, 1.5, 1], \
        [2.6, 0.5, 1], [3.2, -0.7, 0], [3.7, 1.3, 1], [4.3, 3.5, 1], [4.9, 6.8, 0], [5.5, 13.2, 1]])

    with open('save_file.txt', 'a+') as save_file:
        save_file.write("\n")
        save_file.write("###################################################\n")
        save_file.write("###################################################\n")
        save_file.write("iterations = {0}\n".format(ITERATIONS))
        save_file.write("learning_rate = {0}\n".format(LEARNING_RATE))

    for i in xrange(NUM_SEARCH):
        print "##############################################################"
        print "i = {0}".format(i)

        NET = SimpleNeuralNetwork()
        test_accu, lamb, bias, weight = NET.gradient_descent(TRAINING_DATA, ITERATIONS, \
            LEARNING_RATE, TEST_DATA)

        print "i = {0}, test_accu = {1}".format(i, test_accu)
        print "i = {0}, lamb = {1}".format(i, lamb)
        print "i = {0}, bias = {1}".format(i, bias)
        print "i = {0}, weight = {1}".format(i, weight)

        if test_accu > TEST_ACCU_BEST or test_accu >= 0.9:
            TEST_ACCU_BEST = test_accu
            lamb_best = lamb
            bias_best = bias
            weight_best = weight

            with open('save_file.txt', 'a+') as save_file:
                save_file.write("\n")
                save_file.write("###############\n")
                save_file.write("TEST_ACCU_BEST = {0}\n".format(TEST_ACCU_BEST))
                save_file.write("lamb_best = {0}\n".format(lamb_best))
                save_file.write("bias_best = {0}\n".format(bias_best))
                save_file.write("weight_best = {0}\n".format(weight_best))

            print "i = {0}, TEST_ACCU_BEST = {1}".format(i, TEST_ACCU_BEST)
            print "i = {0}, lamb_best = {1}".format(i, lamb_best)
            print "i = {0}, bias_best = {1}".format(i, bias_best)
            print "i = {0}, weight_best = {1}".format(i, weight_best)


    print TEST_ACCU_BEST, lamb_best, bias_best, weight_best
