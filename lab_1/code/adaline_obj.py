import random
import numpy as np
from matplotlib import pyplot as plt

fi_list = [0.7]
bias_list = [1.0, 0.8, 0.05, 0.2]
learning_rate_list = [0.3, 0.2, 0.15, 0.05]
error_treshold_list = [0.7, 0.5, 0.4, 0.1]

class Perceptron(object):
    def __init__(self):
        self.weights = []
        self.x_list = []
        self.y_list = []
        self.epochs = []
        self.learning_rate = 0.1
        self.fi = 0.7
        self.bias = 1.0
        self.accepted_error = 0.3

    def init_class(self, i, points_num, function_type='and'):
        self.y_list.clear()
        self.x_list.clear()
        self.weights.clear()
        self.bias = bias_list[i]
        self.fi = fi_list[0]
        self.learning_rate = learning_rate_list[i]
        self.accepted_error = error_treshold_list[i]
        self.weights = self.generate_weights(self.bias)
        x_copy = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for xp in x_copy:
            self.generate_points(points_num, xp[0], xp[1], function_type)
        print(f"Bias {self.bias} FI {self.fi} Learning_rate {self.learning_rate} Error Treshold {self.accepted_error}")
        print(self.x_list)
        print(self.y_list)
        print(self.weights)
        print("------------------------------------------------------------------------------------")

    def generate_points(self,points_num, x_cord, y_cord, function_type='and'):
        y_to_add = 1
        if function_type == 'and':
            if x_cord != 1 or y_cord != 1:
                y_to_add = 0
        elif function_type == 'or':
            if x_cord == 0 and y_cord == 0:
                y_to_add = 0

        for i in range(points_num):
            if x_cord > 0 and y_cord > 0:
                new_point = (random.uniform(x_cord - 0.07, x_cord + 0.07), random.uniform(y_cord - 0.07, y_cord + 0.07))
            elif x_cord > 0:
                new_point = (random.uniform(x_cord - 0.07, x_cord + 0.07), random.uniform(0, y_cord + 0.07))
            elif y_cord > 0:
                new_point = (random.uniform(0, x_cord + 0.07), random.uniform(y_cord - 0.07, y_cord + 0.07))
            else:
                new_point = (random.uniform(0, x_cord + 0.07), random.uniform(0, y_cord + 0.07))
            self.x_list.append(new_point)
            self.y_list.append(y_to_add)

    def generate_weights(self,bias_val=1.0, num_of_weights=2):
        return [random.uniform(-bias_val, bias_val) for _ in range(num_of_weights)]

    def unipolar_res(self, x, fi=0.1):
        return 1.0 if x > fi else 0.0

    def bipolar_res(self, x, fi=0.1):
        return 1.0 if x > fi else -1.0

    def activation_val(self,x):
        value_sum = 0
        for i in range(len(self.weights)):
            value_sum +=x[i] * self.weights[i]
        a = self.unipolar_res(value_sum)
        return a

    def perceptron_learn(self):
        error_occurred = True
        epochs = 0
        while error_occurred or epochs > 1000:
            error_occurred = False
            for i in range(len(self.x_list)):
                res = self.activation_val(self.x_list[i])
                error_val = self.y_list[i] - res
                if error_val > 0:
                    error_occurred = True
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] + self.learning_rate * error_val * self.x_list[i][j]
            epochs +=1
            print(f"Epoch {epochs}")
            print(self.weights)


if __name__ == '__main__':
    # for i in range(4):
    #     init_class(i, 20)
    #     adaline_learn()
    perceptron = Perceptron()
    perceptron.init_class(0, 20)
    perceptron.perceptron_learn()
    lst = [list(t) for t in zip(*perceptron.x_list)]
    zz = [0,1]
    ax = plt.scatter(lst[0], lst[1])
    plt.plot([-2, -1, 0, 1, 2], [((-perceptron.weights[0] / perceptron.weights[2]) / (perceptron.weights[0] / perceptron.weights[1])) * x + (-perceptron.weights[0] / perceptron.weights[2]) for x in [-2, -1, 0, 1, 2]])
    plt.title("ADALINE Errors")
    plt.show()
    # perceptron_learn()
    # val = activation_value((0.999, 0.987), w)
    # res = unipolar_result(val, fi)
    # print(res)