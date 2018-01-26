import numpy as np
import matplotlib.pyplot as plt
from layers import *
from lib import *
import copy
import random

def Plot_weights_as_Image(w, title):
    weights = w[1:]
    image = np.reshape(weights, (28, 28))
    plt.title(title, fontsize=10)
    plt.imshow(image, cmap='gray')

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
            features.append(float(tokens[i])/255.0)
        curriculum_data.append((features, float(tokens[1]), difficulty))
    return curriculum_data    

def sort_curriculum_data(curriculum_data, sort_by):
    data = copy.deepcopy(curriculum_data)
    data.sort(key=lambda x: x[2])
    if sort_by == "hard":
        data.reverse()
    elif sort_by == "random":
        random.shuffle(data)
    elif sort_by != "easy":
        print("unrecognized sort_by option", sort_by)
        return 1/0

    X = [features for (features, label, difficulty) in data]
    Y = [label for (features, label, difficulty) in data]
    Y = numpy.array(Y)
    X = numpy.array(X)
    return X, Y



def main():
    curriculum_data = load_curriculum_data("curriculum_training_set")
    training_x, training_y = sort_curriculum_data(curriculum_data, "random")
    print training_x[0]
    Plot_weights_as_Image(training_x[0], "debug")
main()