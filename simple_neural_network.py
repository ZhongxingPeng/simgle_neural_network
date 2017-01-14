"""In this script, we implement a simple neural network"""
import math
import random
import numpy as np


class SimpleNeuralNetwork(object):
    """Implement a simple neural network"""

    def __init__(self):
        # Set the initial values for three parameters in our network
        self.lamb = 1.21463071343
        self.bias = 0.971014765329
        self.weight = 0.13225856384


    def feedforward(self, coordinate_x, coordinate_y):
        """feedforward through the network"""

        return sigmoid(coordinate_y - self.lamb * math.pow((self.weight * coordinate_x + \
            self.bias), 2))


    def loss_fun(self, training_data):
        """Return the value of loss function"""

        loss = 0.0
        for coordinate_x, coordinate_y, label in training_data:
            weighted_z = coordinate_y - self.lamb * math.pow((self.weight * coordinate_x + \
                self.bias), 2)
            loss = loss - label * np.log(sigmoid(weighted_z)) - (1-label) * \
                np.log(1 - sigmoid(weighted_z)) # pylint: disable=no-member
        return loss


    def gradient_descent(self, training_data, epochs, eta, test_data):
        """Carry on gradient descent on training data to update parameters"""

        num_test = len(test_data)
        num_training = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            self.update_params(training_data, eta)
            print "Epoch {0}: lambda = {1}, bias = {2}, weight = {3}".format(j, self.lamb, \
                self.bias, self.weight)
            print "Loss: {0}".format(self.loss_fun(training_data))
            print "Training Accu = {0}, Test Accu = {1}".format(1.0 * self.evaluate(
                training_data)/num_training, 1.0 * self.evaluate(test_data)/num_test)


    def update_params(self, training_data, eta):
        """Update parameters in the neural network"""

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
        """Carry on back propagation"""

        # feedforward through the network
        temp_u1 = coordinate_x * self.weight
        temp_u2 = temp_u1 + self.bias
        temp_u3 = math.pow(temp_u2, 2)
        temp_u4 = temp_u3 * self.lamb
        weighted_z = coordinate_y - temp_u4

        # backward to obtain gradient of every parameter
        temp = (label - sigmoid(weighted_z))
        nabla_lamb = temp * temp_u1
        nabla_bias = temp * self.lamb * 2.0 * temp_u2
        nabla_weight = nabla_bias * coordinate_x

        return (nabla_lamb, nabla_bias, nabla_weight)


    def evaluate(self, test_data):
        """Evaluation the neural network"""

        num_correct_predict = 0
        test_results = [(self.feedforward(x, y), label) for (x, y, label) in test_data]

        for predict_prob, label in test_results:
            # Compute the distance between label and predicted label
            # We consider a correct prediction if the distance is smaller than 0.5
            if abs(label - predict_prob) < 0.5:
                num_correct_predict += 1
        return num_correct_predict


def sigmoid(weighted_z):
    """Sigmoid function"""
    return 1.0/(1.0+np.exp(-weighted_z)) # pylint: disable=no-member


if __name__ == '__main__':

    ITERATIONS = 2000000 # Number of iterations for gradient descent
    LEARNING_RATE = 0.001 # Learning rate of gradient descent

    # Training data: in each element, [coordinate x, coordinate y, label]
    TRAINING_DATA = np.array([[-0.5, 12, 0], [0.5, 13.2, 1], [0.8, 8, 0], [1, 9, 1], \
        [1.3, 6.5, 1], [1.5, 5, 1], [1.7, 3, 0], [2.0, 1.5, 0], [2.2, 2, 1], [2.5, 1, 1], \
        [3, -1.3, 0], [3.3, 0.5, 1], [3.5, 1.5, 1], [3.8, 1.1, 0], [4.1, 2.8, 1], [4.5, 5.1, 1], \
        [4.9, 6.8, 0], [5.3, 11.2, 1], [5.7, 14.1, 0], [6.2, 21.2, 1]])

    # Test data
    TEST_DATA = np.array([[0.7, 11.1, 1], [1.3, 5.2, 0], [1.7, 3.0, 0], [2.2, 1.5, 1], \
        [2.6, 0.5, 1], [3.2, -0.7, 0], [3.7, 1.3, 1], [4.3, 3.5, 1], [4.9, 6.8, 0], [5.5, 13.2, 1]])

    # Initialize a network
    NET = SimpleNeuralNetwork()
    # Start to train the network
    NET.gradient_descent(TRAINING_DATA, ITERATIONS, LEARNING_RATE, TEST_DATA)
