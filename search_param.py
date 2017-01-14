import random
import numpy as np


training_data = np.array([[-0.5,12,0],[0.5,13.2,1],[0.8,8,0],[1,9,1],
    [1.3,6.5,1], [1.5,5,1],[1.7,3,0],[2.0,1.5,0],[2.2,2,1], [2.5,1,1],
    [3,-1.3,0], [3.3,0.5,1],[3.5,1.5,1],[3.8,1.1,0],[4.1,2.8,1],[4.5,5.1,1],
    [4.9,6.8,0], [5.3,11.2,1],[5.7,14.1,0],[6.2,21.2,1]])
test_data = np.array([[0.7,11.1,1],[1.3,5.2,0],[1.7,3.0,0],[2.2,1.5,1], 
    [2.6,0.5,1],[3.2,-0.7,0],[3.7,1.3,1],[4.3,3.5,1],[4.9,6.8,0], [5.5,13.2,1]])


class Network(object):

    def __init__(self):
        self.lamb = np.random.randn()
        self.bias = np.random.randn()
        self.weight = np.random.randn()

        self.lamb_init = self.lamb
        self.bias_init = self.bias
        self.weight_init = self.weight

        print "self.lamb = ", self.lamb
        print "self.bias = ", self.bias
        print "self.weight = ", self.weight
 

    def feedforward(self, x, y):
        return sigmoid(y - self.lamb * (self.weight * x + self.bias) * \
            (self.weight * x + self.bias))

    def loss_fun(self, training_data):
        loss = 0.0
        for x, y, label in training_data:
            z = y - self.lamb * (self.weight * x + self.bias) * (self.weight * x + self.bias)
            loss = loss - label * np.log(sigmoid(z)) - (1-label) * np.log(1 - sigmoid(z))
        return loss

    def SGD(self, training_data, epochs, eta, test_data=None):
        if test_data is not None: 
            num_test = len(test_data)
        num_training = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            self.update_params(training_data,eta)
            # if test_data is not None:
            #     print "Epoch {0}: lamb = {1}, bias = {2}, weight = {3}".format(j, self.lamb, self.bias, self.weight)
            #     print "Loss: {0}".format(self.loss_fun(training_data))
            #     print "Training Accu = {0}/{1}, Test Accu = {2}/{3}".format(self.evaluate(training_data), num_training, self.evaluate(test_data), num_test)
            # else:
            #     print "Epoch {0} complete".format(j)
        test_accu = 1.0 * self.evaluate(test_data)/num_test
        return test_accu, self.lamb_init, self.bias_init, self.weight_init 
 

    def update_params(self, training_data, eta):
        nabla_lamb   = 0.0
        nabla_bias   = 0.0
        nabla_weight = 0.0
        for x, y, label in training_data:
            delta_nabla_lamb, delta_nabla_bias, delta_nabla_weight = self.backprop(x, y, label)
            nabla_lamb   = nabla_lamb   + delta_nabla_lamb
            nabla_bias   = nabla_bias   + delta_nabla_bias
            nabla_weight = nabla_weight + delta_nabla_weight
        self.lamb   = self.lamb   - (eta/len(training_data)) * nabla_lamb
        self.bias   = self.bias   - (eta/len(training_data)) * nabla_bias            
        self.weight = self.weight - (eta/len(training_data)) * nabla_weight
 

    def backprop(self, x, y, label):
        # feedforward
        u1 = x * self.weight
        u2 = u1 + self.bias
        u3 = u2 * u2
        u4 = u3 * self.lamb
        z  = y - u4

        # backward pass
        temp = (label - sigmoid(z))
        nabla_lamb = temp * u1
        nabla_bias = temp * self.lamb * 2 * u2
        nabla_weight = nabla_bias * x

        return (nabla_lamb, nabla_bias, nabla_weight)
 
    def evaluate(self, test_data):
        num_correct_predict = 0
        test_results = [(self.feedforward(x,y), label) for (x, y, label) in test_data]
        for predict_prob, label in test_results:
            if abs(label - predict_prob) < 0.5:
                num_correct_predict += 1 
        return num_correct_predict


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


if __name__ == '__main__':
    lamb_best = 0.0
    bias_best = 0.0
    weight_best = 0.0
    test_accu_best = 0.0
    iterations = 100000
    learning_rate = 0.001

    save_file = open('save_file.txt', 'a+')
    with open('save_file.txt', 'a+') as save_file:
        save_file.write("\n")
        save_file.write("###################################################\n")
        save_file.write("###################################################\n")
        save_file.write("iterations = {0}\n".format(iterations))
        save_file.write("learning_rate = {0}\n".format(learning_rate))
    save_file.close

    for i in xrange(20000000000):
        print "##############################################################"
        print "i = {0}".format(i)
        net = Network()
        test_accu, lamb, bias, weight = net.SGD(training_data, iterations, 
            learning_rate, test_data=test_data)
        print "i = {0}, test_accu = {1}".format(i, test_accu)
        print "i = {0}, lamb = {1}".format(i, lamb)
        print "i = {0}, bias = {1}".format(i, bias)
        print "i = {0}, weight = {1}".format(i, weight)

        if test_accu > test_accu_best:
            test_accu_best = test_accu
            lamb_best = lamb
            bias_best = bias
            weight_best = weight

            with open('save_file.txt', 'a+') as save_file:
                save_file.write("\n")
                save_file.write("###############\n")
                save_file.write("test_accu_best = {0}\n".format(test_accu_best))
                save_file.write("lamb_best = {0}\n".format(lamb_best))
                save_file.write("bias_best = {0}\n".format(bias_best))
                save_file.write("weight_best = {0}\n".format(weight_best))
            save_file.close

            print "i = {0}, test_accu_best = {1}".format(i, test_accu_best)
            print "i = {0}, lamb_best = {1}".format(i, lamb_best)
            print "i = {0}, bias_best = {1}".format(i, bias_best)
            print "i = {0}, weight_best = {1}".format(i, weight_best)


    print test_accu_best, lamb_best, bias_best, weight_best
