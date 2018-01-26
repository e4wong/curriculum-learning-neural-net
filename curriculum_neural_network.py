from layers import *
from lib import *
import copy
import random
import matplotlib.pyplot as plt

def predict(layers, X):
    X_next = X
    for layer in layers:
        X_next = layer.forward_prop(X_next)
    Y_pred = unhot(X_next)
    return Y_pred

def get_loss(layers, X, Y_one_hot):
    X_next = X
    for layer in layers:
        X_next = layer.forward_prop(X_next)
    Y_pred = X_next
    return layers[-1].logistic_loss(Y_one_hot, Y_pred)

def error(layers, X, Y):
    """ Calculate error on the given data. """
    Y_pred = predict(layers, X)
    error = Y_pred != Y
    return numpy.mean(error)

def load_curriculum_data(filename):
    f = open(filename, "r")
    curriculum_data = []
    for line in f:
        tokens = line.split()
        difficulty = float(tokens[0])
        features = []        
        for i in range(2, len(tokens)):
            features.append(float(tokens[i]))
        curriculum_data.append((features, float(tokens[1]), difficulty))
    return curriculum_data    

def setup(layers, X, Y):
    next_shape = X.shape
    for layer in layers:
        layer.setup(next_shape)
        next_shape = layer.output_shape(next_shape)
    if next_shape != Y.shape:
        print "We messed up the dimensions"
        return 1/0

def sort_curriculum_data(curriculum_data, sort_by):
    data = copy.deepcopy(curriculum_data)
    data.sort(key=lambda x: x[2])
    if sort_by == "hard":
        data.reverse()
    elif sort_by == "random":
        random.shuffle(data)
    elif sort_by != "easy":
        print "unrecognized sort_by option", sort_by
        return 1/0
    X = [features for (features, label, difficulty) in curriculum_data]
    Y = [label for (features, label, difficulty) in curriculum_data]
    Y = numpy.array(Y)
    X = numpy.array(X)
    return X, Y

def stepsize_function(k, iteration, T):
    return float(k)/(1.0 + iteration/T)

def train(layers, training_x, training_y, test_x, test_y):
    y_one_hot = encode_one_hot(training_y)
    test_y_one_hot = encode_one_hot(test_y)

    setup(layers, training_x, y_one_hot)

    stepsize = 0.01
    training_size = len(training_x)
    training_err = []
    training_loss = []
    test_err = []
    test_loss = []

    training_err.append(error(layers, training_x, training_y))
    test_err.append(error(layers, test_x, test_y))
    training_loss.append(get_loss(layers, training_x, y_one_hot))
    test_loss.append(get_loss(layers, test_x, test_y_one_hot))

    training_size = len(training_x)
    for i in range(0, 1):
        # not using whole training set
        for batch_number in range(0, training_size/50):
            batch_x = training_x[batch_number * 10 : (batch_number + 1) * 10]
            batch_y = y_one_hot[batch_number * 10 : (batch_number + 1) * 10]
            X_next = batch_x

            for layer in layers:
                X_next = layer.forward_prop(X_next)
            Y_pred = X_next
            next_grad = layers[-1].input_gradient(batch_y, Y_pred)

            for layer in reversed(layers[:-1]):
                next_grad = layer.back_prop(next_grad)

            for layer in layers:
                if isinstance(layer, LinearLayer):
                    layer.W -= stepsize_function(stepsize, batch_number, 100) * layer.dW
            training_err.append(error(layers, training_x, training_y))
            test_err.append(error(layers, test_x, test_y))
            training_loss.append(get_loss(layers, training_x, y_one_hot))
            test_loss.append(get_loss(layers, test_x, test_y_one_hot))
    return training_err, training_loss, test_err, test_loss

def curriculum(curriculum_data, sort_by, test_x, test_y):
    X, Y = sort_curriculum_data(curriculum_data, "hard")
    layers=[LinearLayer(64), ActivationLayer(logistic, logistic_derivative), LinearLayer(10), LogisticRegressionLayer()]
    return train(layers, X, Y, test_x, test_y)

def main():
    curriculum_data = load_curriculum_data("curriculum_training_set")
    test_x, test_y = load_data("test_set")
    print "Running Easy -> Hard"
    easy_training_err, easy_training_loss, easy_test_err, easy_test_loss = curriculum(curriculum_data, "easy", test_x, test_y)
    print "Running random"
    random_training_err, random_training_loss, random_test_err, random_test_loss = curriculum(curriculum_data, "random", test_x, test_y)
    print "Running Hard -> Easy"
    hard_training_err, hard_training_loss, hard_test_err, hard_test_loss = curriculum(curriculum_data, "hard", test_x, test_y)
    f_0 = plt.figure(0)
    f_0.canvas.set_window_title("Training Error")
    plt.plot(easy_training_err)
    plt.plot(random_training_err)
    plt.plot(hard_training_err)
    plt.legend(['Easy Examples First', 'Random Examples', 'Hard Examples First'], loc='upper left')
    plt.ylabel('Error Rate')
    plt.show()    


main()