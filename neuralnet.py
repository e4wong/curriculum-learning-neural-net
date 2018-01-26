from layers import *
from lib import *

def predict(layers, X):
    X_next = X
    for layer in layers:
        X_next = layer.forward_prop(X_next)
    Y_pred = unhot(X_next)
    return Y_pred


def error(layers, X, Y):
    """ Calculate error on the given data. """
    Y_pred = predict(layers, X)
    error = Y_pred != Y
    return numpy.mean(error)

def setup(layers, X, Y):
    next_shape = X.shape
    for layer in layers:
        print next_shape
        layer.setup(next_shape)
        next_shape = layer.output_shape(next_shape)
    print next_shape, Y.shape
    if next_shape != Y.shape:
        print "We messed up the dimensions"
        return 1/0

def train(layers, training_x, training_y, test_x, test_y):
    y_one_hot = encode_one_hot(training_y)
    test_y_one_hot = encode_one_hot(test_y)

    setup(layers, training_x, y_one_hot)
    stepsize = 0.1

    print error(layers, training_x, training_y) 
    training_size = len(training_x)
    for i in range(0, 100):
        for batch_number in range(0, 9):
            batch_x = training_x[(batch_number * training_size)/10 : (batch_number + 1) * training_size/10]
            batch_y = y_one_hot[(batch_number * training_size)/10 : (batch_number + 1) * training_size/10]

            X_next = batch_x
            for layer in layers:
                X_next = layer.forward_prop(X_next)
            Y_pred = X_next

            next_grad = layers[-1].input_gradient(batch_y, Y_pred)
            for layer in reversed(layers[:-1]):
                next_grad = layer.back_prop(next_grad)

            for layer in layers:
                if isinstance(layer, LinearLayer):
                    layer.W -= stepsize * layer.dW
        print "training", error(layers, training_x, training_y), get_loss(layers, training_x, y_one_hot)
    return  

def curriculum(layers, training_x, training_y):
    print "final training", error(layers, training_x, training_y)
    curriculum_labeled = []
    X_next = training_x
    for layer in layers:
        X_next = layer.forward_prop(X_next)
    Y_pred = X_next

    for i in range(0, len(Y_pred)):
        probabilities = Y_pred[i]
        difficulty = 1 - probabilities[int(training_y[i])]
        curriculum_labeled.append((training_x[i], difficulty))

    return curriculum_labeled

def output_curriculum_training_set(curriculum_training_set, training_y):
    f = open("curriculum_training_set", 'w')
    for i in range(0, len(curriculum_training_set)):
        s = ""
        (sample, difficulty) = curriculum_training_set[i]
        label = training_y[i]
        s += str(difficulty) + " " + str(label) + " "
        for elem in sample:
            s += str(elem) + " "
        s += "\n"
        f.write(s)

def get_loss(layers, X, Y_one_hot):
    X_next = X
    for layer in layers:
        X_next = layer.forward_prop(X_next)
    Y_pred = X_next
    return layers[-1].logistic_loss(Y_one_hot, Y_pred)

def main():
    training_x, training_y = load_data("training_set")
    test_x, test_y = load_data("test_set")

    layers=[LinearLayer(64), ActivationLayer(logistic, logistic_derivative), LinearLayer(10), LogisticRegressionLayer()]
    train(layers, training_x, training_y, test_x, test_y)   
    curriculum_training_set = curriculum(layers, training_x, training_y)
    print curriculum_training_set[0]
    print training_x[0] == curriculum_training_set[0][0]
    print training_y[0]
    #output_curriculum_training_set(curriculum_training_set, training_y)

main()