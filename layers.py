import numpy

class ActivationLayer():

    def __init__(self, function, function_derivative):
        self.activation_function = function
        self.activation_function_derivative = function_derivative

    def setup(self, input_shape):
        return
        
    def forward_prop(self, input):
        self.last_input = input
        return self.activation_function(input)

    def back_prop(self, output_grad):
        return output_grad*self.activation_function_derivative(self.last_input)

    def output_shape(self, input_shape):
        return input_shape

class LinearLayer():
    
    def __init__(self, num_output_nodes):
        self.num_output_nodes = num_output_nodes
        self.rng = numpy.random.RandomState()
        return

    def output_shape(self, input_shape):
        return (input_shape[0], self.num_output_nodes)

    def setup(self, input_shape):
        num_inputs = input_shape[1]
        weight_shape = (num_inputs, self.num_output_nodes)
        self.W = self.rng.normal(size=weight_shape, scale=0.1)
        
    def forward_prop(self, input):
        self.last_input = input
        return numpy.dot(input, self.W)

    def back_prop(self, gradient):
        n = gradient.shape[0]
        self.dW = numpy.dot(self.last_input.T, gradient)/n
        return numpy.dot(gradient, self.W.T)

class LogisticRegressionLayer():

    def setup(self, input_shape):
        return

    def forward_prop(self, input):
        e = numpy.exp(input - numpy.amax(input, axis=1, keepdims=True))
        return e/numpy.sum(e, axis=1, keepdims=True)
    
    def input_gradient(self, y, y_pred):

        if y.shape != y_pred.shape:
            print "Error y and y_pred not same shape"
            print y.shape, y_pred.shape
            return 1/0
        return -(y - y_pred)

    def logistic_loss(self, y, y_pred):
        loss = -numpy.sum(y * numpy.log(y_pred))
        return loss / y.shape[0]      

    def output_shape(self, input_shape):
        return input_shape
