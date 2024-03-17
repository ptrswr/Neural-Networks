from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import math
import numpy as np


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
    return (x_train, y_train), (x_test, y_test)


def normalize_data(images, labels):
    images = np.array(images, dtype='float32')
    images /= 255
    labels = to_categorical(labels)
    return images, labels


class Mlp:
    def __init__(self, data_shape):
        self.layers = []
        self.weights = []
        self.biases = []
        self.functions = []
        self.data_shape = data_shape
        self.learning_rate = 0.1

    def add_layer(self, number_of_nodes, function="sigmoid"):
        # number of nodes per each layer
        if len(self.layers) == 0:
            layer_shape = self.data_shape[1]
            bias_shape = self.data_shape[0]
        else:
            layer_shape = self.weights[-1].shape[0]
            bias_shape = self.data_shape[1]

        self.layers.append(number_of_nodes)
        layer_weights = np.random.normal(scale=1, size=(number_of_nodes, layer_shape))
        self.weights.append(layer_weights)
        self.biases.append(np.ones((number_of_nodes, bias_shape)))

        self.functions.append(function)


    @staticmethod
    def softmax(outputs):
        result = []
        for x in outputs:
            total_sum = np.sum(x, axis=0) / 1000
            e_to_the_y_j = np.exp(total_sum)
            result.append(e_to_the_y_j / np.sum(e_to_the_y_j, axis=0))
        return np.array(result)


    @staticmethod
    def tanh(x):
        tg = np.vectorize(lambda y: (2 / (1 + math.exp(-2 * y)) - 1))
        return tg(x)

    @staticmethod
    def relu(x):
        re = np.vectorize(lambda y: max(0, y))
        return re(x)

    @staticmethod
    def sigmoid(x):
        sig = np.vectorize(lambda y: (1 / (1 + math.exp(-y))))
        return sig(x)

    @staticmethod
    def derivative(x, function):
        if function == "sigmoid":
            return np.multiply(x, (1 - x))
        elif function == "soft_plus":
            return Mlp.sigmoid(x)
        elif function == "relu":
            d_relu = np.vectorize(lambda y: 1 if y > 0 else 0)
            return d_relu(x)

    @staticmethod
    def activate(x, function):
        if function == "sigmoid":
            return Mlp.sigmoid(x)
        elif function == "tangens":
            return Mlp.tanh(x)
        elif function == "relu":
            return Mlp.relu(x)

    def feed_forward(self, inputs):
        outputs_layer = [np.array(inputs)]
        # -1 because we do not perform last multiplication
        for i in range(len(self.layers)):
            temp_matrix = np.dot(outputs_layer[-1], self.weights[i].T)
            for element in temp_matrix:
                element += self.biases[i].T
            outputs_layer.append(Mlp.activate(temp_matrix, self.functions[i]))

        return outputs_layer

    def train(self, x_train, y_train):

        layer_output = self.feed_forward(x_train)
        softed = Mlp.softmax(layer_output[-1])

        errors = [np.subtract(y_train, softed)]
        print(errors)
        for i in range(len(self.weights) - 1):
            errors.insert(0, np.dot(self.weights[-1 - i].T, errors[0]))

        for i in range(len(self.weights)):
            # Calculate gradient and weight correction
            gradient = np.multiply(errors[-1 - i], Mlp.derivative(layer_output[-1 - i], self.functions[-1 - i]))
            gradient *= self.learning_rate
            self.biases[-1 - i] += gradient
            delta_w = np.dot(gradient, layer_output[-2 - i].T)
            self.weights[-1 - i] += delta_w
        print(softed.shape)
        print(softed[:3])
        print(np.sum(softed))


if __name__ == '__main__':
    dict1 = {"username": "name1", "client_name": "client1"}
    dict2 = {"username": "name2", "client_name": "client2"}
    dict3 = {"username": "name3", "client_name": "client3"}
    dict4 = {"username": "name4", "client_name": "client4"}

    ll = [dict1,dict2,dict3,dict4]
    message = ("Informa \r\n")
    for l in ll:
        message += (f"new user: {l['username']} nad clint {l['client_name']} \r\n")

    message += "koniec"

    print(message)
