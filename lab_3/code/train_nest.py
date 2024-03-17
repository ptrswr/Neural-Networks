import math

from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split


class Mlp:
    def __init__(self, weights_init="Xavier"):
        self.layer_sizes = []
        self.layer_functions = []
        self.weights = []
        self.biases = []
        self.weights_init = "Xavier"


        self.activated = []
        self.not_activated = []
        self.prev_weights_updates = []

    def add_layer(self, size, function):
        self.layer_sizes.append(size)
        self.layer_functions.append(function)

    def add_output_layer(self):
        self.layer_sizes.append(10)
        self.layer_functions.append('softmax')
        self.init_parameters()

    def Xavier(self):
        print("Xavier")
        return [np.random.normal(loc=0, scale=2 / (self.layer_sizes[i] + self.layer_sizes[i + 1]),
                                 size=(self.layer_sizes[i + 1], self.layer_sizes[i]))
                for i in range(len(self.layer_sizes) - 1)]

    def He(self):
        print("He")
        return [np.random.normal(loc=0, scale=2 / self.layer_sizes[i + 1],
                                 size=(self.layer_sizes[i + 1], self.layer_sizes[i]))
                for i in range(len(self.layer_sizes) - 1)]

    def Random(self):
        print("Random")
        return [np.random.normal(loc=0, scale=1, size=(self.layer_sizes[i + 1], self.layer_sizes[i]))
                for i in range(len(self.layer_sizes) - 1)]

    def choose_init_weights(self):
        if self.weights_init == "He":
            return self.He()
        elif self.weights_init == "Xavier":
            return self.Xavier()
        else:
            return self.Random()

    def update_net_parameters(self, weight_updates, bias_updates, l_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= l_rate * weight_updates[i]
            self.biases[i] -= l_rate * bias_updates[i]

    def update_net_parameters_batch(self, weight_updates, bias_updates, num_of_elems_in_batch, l_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= l_rate * weight_updates[i] / num_of_elems_in_batch
            self.biases[i] -= l_rate * bias_updates[i] / num_of_elems_in_batch

    def update_params_batch_not_proceeded(self, weight_updates, bias_updates, batch_size, l_rate):
        for i in range(len(self.weights)):
            update = l_rate * weight_updates[i] / batch_size
            self.weights[i] -= update
            self.prev_weights_updates[i] = update
            self.biases[i] -= l_rate * bias_updates[i] / batch_size
        self.prev_weights_updates = np.array(self.prev_weights_updates)

    def update_params_batch(self, weight_updates, bias_updates, batch_size, l_rate, gamma):
        for i in range(len(self.weights)):
            update_nesterov = gamma * self.prev_weights_updates[i] + l_rate * weight_updates[i] / batch_size
            self.weights[i] -= update_nesterov
            self.biases[i] -= l_rate * bias_updates[i] / batch_size
            self.prev_weights_updates[i] = update_nesterov

    def init_parameters(self):
        self.weights = []
        self.biases = []
        self.weights = self.choose_init_weights()
        for i in range(len(self.layer_sizes) - 1):
            self.biases.append(np.ones(self.layer_sizes[i + 1]))
        self.prev_weights_updates = [None] * len(self.weights)

    def rev_arrs_back_prop(self):
        return self.not_activated[::-1], self.activated[::-1], self.weights[::-1], self.layer_functions[::-1]

    @staticmethod
    def load_data():
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        x = np.array(x, dtype=np.float64)
        x /= 255
        y = to_categorical(y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        print(f"x train shape: {x_train.shape}")
        print(f"y train shape: {y_train.shape}")
        print(f"x test shape: {x_test.shape}")
        print(f"y test shape: {y_test.shape}")

        return x_train, x_test, y_train, y_test

    @staticmethod
    def activate(x, function):
        if function == 'sigmoid':
            return Mlp.sigmoid(x)
        if function == 'tanh':
            return Mlp.tanh(x)
        if function == 'relu':
            return Mlp.relu(x)
        if function == 'softmax':
            return Mlp.softmax(x)

    @staticmethod
    def derivative(x, function):
        if function == 'sigmoid':
            return Mlp.sigmoid_dev(x)
        if function == 'tanh':
            return Mlp.tanh_dev(x)
        if function == 'relu':
            return Mlp.relu_dev(x)
        if function == 'softmax':
            return Mlp.softmax_dev(x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_dev(x):
        f = Mlp.sigmoid(x)
        return f * (1 - f)

    @staticmethod
    def tanh(x):
        return (2 / (1 + np.exp(-2 * x))) + 1

    @staticmethod
    def tanh_dev(x):
        return 1 - (Mlp.tanh(x) ** 2)

    @staticmethod
    def relu(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def relu_dev(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        to_exp = np.exp(x - x.max())
        return to_exp / np.sum(to_exp, axis=0)

    @staticmethod
    def softmax_dev(x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

    def backward_propagation(self, y_train, out, gamma):
        before_activation, activated, weights, layer_functions = self.rev_arrs_back_prop()
        prev_weights_updates = self.prev_weights_updates[::-1]
        # processing softmaxed layer
        error = (out - y_train) * Mlp.derivative(before_activation[0], layer_functions[0])
        # Hadamard prod
        weights_modifiers = [np.outer(error, activated[1])]
        biases_modifiers = [error]

        weights_mod = weights if prev_weights_updates[0] is None else np.array(
            weights) - prev_weights_updates * gamma

        # backward prop of other layers with derivative of their functions
        # we are ommitting first (which is now last) layer that was already processed before loop
        for i in range(len(self.weights) - 1):
            error = np.dot(weights_mod[i].T, error) * \
                    Mlp.derivative(before_activation[i + 1], layer_functions[i + 1])
            biases_modifiers.append(error)
            weights_modifiers.append(np.outer(error, activated[i + 2]))

        return weights_modifiers[::-1], biases_modifiers[::-1]

    def feed_forward(self, x_train):
        # dot of x and w after activation
        self.activated = [x_train]
        # dot of x and w before activation
        self.not_activated = []
        for i in range(len(self.layer_sizes) - 1):
            mul_output = np.dot(np.array(self.weights[i]), self.activated[-1]) + self.biases[i]
            self.not_activated.append(mul_output)
            output_activated = Mlp.activate(mul_output, self.layer_functions[i])
            self.activated.append(output_activated)

        return self.activated[-1]

    def train(self, x_train, y_train, x_test, y_test, epochs=10, l_rate=0.05):
        for i in range(epochs):
            for x, y in zip(x_train, y_train):
                x = np.reshape(x, x.shape[0])
                softed_output = self.feed_forward(x)
                weight_updates, bias_updates = self.backward_propagation(y, softed_output)
                self.update_net_parameters(weight_updates, bias_updates, l_rate=l_rate)
            print(f"Epoch {i}")
            accuracy = self.predict(x_test, y_test)
            print(f'Accuracy: {accuracy * 100}')



    def train_batch_v2(self, x_train, y_train, x_test, y_test, num_of_batches=2000, epochs=10, l_rate=0.5, gamma=0.8):
        batches_x = np.array_split(x_train, num_of_batches)
        batches_y = np.array_split(y_train, num_of_batches)
        min_err = math.inf
        weight_cp = np.copy(self.weights)
        bias_cp = np.copy(self.biases)
        for i in range(epochs):
            first_update_to_make = False
            for x_b, y_b in zip(batches_x, batches_y):
                weight_updates = None
                bias_updates = None
                for x, y in zip(x_b, y_b):
                    x = np.reshape(x, x.shape[0])
                    softed_output = self.feed_forward(x)
                    weight_updates_val, bias_updates_val = self.backward_propagation(y, softed_output, gamma)
                    if weight_updates is None and bias_updates is None:
                        weight_updates = weight_updates_val
                        bias_updates = bias_updates_val
                    else:
                        for j in range(len(weight_updates_val)):
                            bias_updates[j] += bias_updates_val[j]
                            weight_updates[j] += weight_updates_val[j]
                if first_update_to_make:
                    self.update_params_batch(weight_updates, bias_updates, len(x_b), l_rate=l_rate, gamma=gamma)
                else:
                    self.update_params_batch_not_proceeded(weight_updates, bias_updates, len(x_b), l_rate=l_rate)
                if not first_update_to_make:
                    first_update_to_make = True
            print(f"Epoch {i}")
            accuracy = self.predict(x_test, y_test)
            print(f'Accuracy: {accuracy * 100}')
            if 1 - accuracy < min_err:
                min_err = 1 - accuracy
                weight_cp = np.copy(self.weights)
                bias_cp = np.copy(self.biases)
            else:
                print("weight reverb")
                self.weights = weight_cp
                self.biases = bias_cp

    def predict(self, x_test, y_test):
        predictions = []
        for x, y in zip(x_test, y_test):
            output = self.feed_forward(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        return np.mean(predictions)

def main():
    x_train, x_test, y_train, y_test = Mlp.load_data()
    l_rates = [0.1, 0.05, 0.01]
    batch_sizes = [1000, 2000, 3000]
    mlp = Mlp()
    #
    mlp.add_layer(784, 'relu')
    mlp.add_layer(128, 'relu')
    mlp.add_layer(64, 'relu')
    mlp.add_output_layer()
    mlp.train_batch_v2(x_train, y_train, x_test, y_test, l_rate=0.5,epochs=10, num_of_batches=2000)



if __name__ == '__main__':
    main()
