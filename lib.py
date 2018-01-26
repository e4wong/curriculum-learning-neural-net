import numpy

def load_data(filename):
    f = open(filename, "r")
    print "Reading from file " + filename
    X = []
    Y = []
    for line in f:
        tokens = line.split()
        features = []
        # this is for the bias feature
        features.append(1.0)

        for i in range(1, len(tokens)):
            features.append(float(tokens[i])/255.0)
        X.append(features)
        Y.append(float(tokens[0]))
    Y = numpy.array(Y)
    X = numpy.array(X)
    print "Done loading data"
    return X, Y

def unhot(one_hot_labels):
    return numpy.argmax(one_hot_labels, axis=-1)

def encode_one_hot(labels):
    classes = numpy.unique(labels)
    n_classes = classes.size
    one_hot_labels = numpy.zeros(labels.shape + (n_classes,))
    for i in range(0, len(one_hot_labels)):
        label = labels[i]
        one_hot_labels[i][int(label)] = 1

    return one_hot_labels




def logistic(x):
    return 1.0/(1.0+numpy.exp(-x))

def logistic_derivative(x):
    s = logistic(x)
    return s*(1-s)