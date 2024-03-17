import math
import random
import datetime
from matplotlib import pyplot as plt

NUM_OF_POINTS =20
POINT_OFFSET = 0.1

x = []
y = []
w = []

weight_lim = 0.2
learning_rate = 0.1
error_treshold = 0.1

is_unipolar = False

weight_lim_list = [1.0, 0.8, 0.5, 0.3,0.1]
learning_rate_list = [0.1, 0.05, 0.001, 0.0001,0.00001]
error_treshold_list = [0.8,0.7, 0.5, 0.4, 0.3,0.29]


def init_class(num_of_points, point_offset, is_unipolar_val=False, weight_lim_val=0.5, learning_rate_val=0.001,
               error_treshold_val=0.52):
    global weight_lim, learning_rate, error_treshold, w, x, y, is_unipolar
    weight_lim = weight_lim_val
    learning_rate = learning_rate_val
    error_treshold = error_treshold_val
    is_unipolar = is_unipolar_val
    w = generate_weights(weight_lim)
    x, y = generate_points(num_of_points, point_offset)
    print(f"weight_lim {weight_lim} Learning_rate {learning_rate} Error Treshold {error_treshold}")
    print("------------------------------------------------------------------------------------")


def unipolar_result(z):
    if z >= 0.001:
        return 1
    return 0


def bipolar_result(z):
    if z >= 0.001:
        return 1
    return -1


def predict(x_list):
    temp = w[0]
    for i in range(len(x_list)):
        temp += x_list[i] * w[i + 1]
    return temp


def generate_points(size, point_lim):
    if is_unipolar:
        zero_treshold = 0
    else:
        zero_treshold = -1
    x_data = []
    y_data = []

    for i in range(size):
        x_data.append((random.uniform(zero_treshold - point_lim, zero_treshold + point_lim),
                       random.uniform(zero_treshold - point_lim, zero_treshold + point_lim)))
        y_data.append(zero_treshold)

        x_data.append((random.uniform(zero_treshold - point_lim, zero_treshold + point_lim),
                       random.uniform(1 - point_lim, 1 + point_lim)))
        y_data.append(zero_treshold)

        x_data.append((random.uniform(1 - point_lim, 1 + point_lim),
                       random.uniform(zero_treshold - point_lim, zero_treshold + point_lim)))
        y_data.append(zero_treshold)

        x_data.append(
            (random.uniform(1 - point_lim, 1 + point_lim), random.uniform(1 - point_lim, 1 + point_lim)))
        y_data.append(1)
    return x_data, y_data


def generate_weights(weight_lim_val=0.2, num_of_weights=2):
    return [1.0] + [random.uniform(-weight_lim_val, weight_lim_val) for _ in range(num_of_weights)]


def calc_error(desired, predicted):
    if is_unipolar:
        return desired - unipolar_result(predicted)
    else:
        return desired - bipolar_result(predicted)


def perceptron_learn():
    error_occurred = True
    epochs = 0
    start = datetime.datetime.now()
    while error_occurred or epochs > 1000:
        error_occurred = False
        epochs += 1
        for i in range(len(x)):
            predicted = predict(x[i])
            error_val = calc_error(y[i], predicted)
            if error_val > 0:
                error_occurred = True
            w[0] = w[0] + learning_rate * error_val
            for j in range(len(w) - 1):
                w[j + 1] = w[j + 1] + learning_rate * error_val * x[i][j]
    end = datetime.datetime.now()
    print(f"Time of calculation {(end - start)}")
    print(f"Epoch {epochs}")


def adaline_learn():
    current_mean_square_error = math.inf
    epochs = 0
    start = datetime.datetime.now()
    while current_mean_square_error > error_treshold or epochs > 1000:
        temp_errors = 0
        epochs += 1
        for i in range(len(x)):
            diff_err = y[i] - predict(x[i])
            squar_err = pow(diff_err, 2)
            temp_errors += squar_err
            w[0] = w[0] + 2 * learning_rate * diff_err
            for j in range(len(w) - 1):
                w[j + 1] = w[j + 1] + (2 * learning_rate * x[i][j] * diff_err)
        current_mean_square_error = temp_errors / len(x)
        # print(current_mean_square_error)
    end = datetime.datetime.now()
    print(f"Error {current_mean_square_error}")
    print(f"Time of calculation {(end - start)}")
    print(f"Epoch {epochs}")


def show_results():
    x_test, y_test = generate_points(20, 0.1)

    correct = 0
    for i in range(len(x_test)):
        if is_unipolar:
            prediction = unipolar_result(predict(x_test[i]))
        else:
            prediction = bipolar_result(predict(x_test[i]))
        if y_test[i] == prediction:
            correct += 1
    print(f"correct: {correct} / {len(x_test)}")

    lst = [list(t) for t in zip(*x_test)]
    plt.scatter(lst[0], lst[1])
    plt.plot([-2, -1, 0, 1, 2], [((-w[0] / w[2]) / (w[0] / w[1])) * x + (-w[0] / w[2]) for x in [-2, -1, 0, 1, 2]])
    plt.title("Predicted separation of classes")
    plt.show()


def conduct_tests_perceptron():
    # for w_val in weight_lim_list:
    #     init_class(NUM_OF_POINTS, POINT_OFFSET, is_unipolar_val=True, weight_lim_val=w_val)
    #     perceptron_learn()
    #     show_results()
    #     print(f"End of test for weight_lim = {w_val} ***************************************\n\n")
    #
    # for l_val in learning_rate_list:
    #     init_class(NUM_OF_POINTS, POINT_OFFSET, is_unipolar_val=True, learning_rate_val=l_val)
    #     perceptron_learn()
    #     show_results()
    #     print(f"End of test for learnig_rate = {l_val} ***************************************\n\n")
    #
    init_class(NUM_OF_POINTS, POINT_OFFSET, is_unipolar_val=True)
    perceptron_learn()
    show_results()
    print(f"End of test for unipolar func ***************************************\n\n")

    init_class(NUM_OF_POINTS, POINT_OFFSET, is_unipolar_val=False)
    perceptron_learn()
    show_results()
    print(f"End of test for unipolar func ***************************************\n\n")

def conduct_tests_adaline():
    # for w_val in weight_lim_list:
    #     init_class(NUM_OF_POINTS, POINT_OFFSET, is_unipolar_val=False, weight_lim_val=w_val)
    #     adaline_learn()
    #     show_results()
    #     print(f"End of test for weight_lim = {w_val} ***************************************\n\n")

    # for l_val in learning_rate_list:
    #     init_class(NUM_OF_POINTS, POINT_OFFSET, is_unipolar_val=False, learning_rate_val=l_val)
    #     adaline_learn()
    #     show_results()
    #     print(f"End of test for learnig_rate = {l_val} ***************************************\n\n")
    #
    for e_val in error_treshold_list:
        init_class(NUM_OF_POINTS, POINT_OFFSET, is_unipolar_val=False, error_treshold_val=e_val)
        adaline_learn()
        show_results()
        print(f"End of test for accepted_error = {e_val} **************************************\n\n")




if __name__ == '__main__':
    init_class(20, 0.1, False,error_treshold_val=0.3, learning_rate_val=0.3)
    # adaline_learn()
    perceptron_learn()
    show_results()
    # w = generate_weights(0.5)
    # perceptron_learn()
    # show_results()
    # conduct_tests_perceptron()
    # conduct_tests_adaline()
