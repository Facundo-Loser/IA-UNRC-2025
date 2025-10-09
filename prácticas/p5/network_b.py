"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
from mnist_loader_b import vectorized_result_binary

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) # cant capas
        self.sizes = sizes
        # generate Gaussian distributions with mean 0 and standard deviation 1:
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]                      # Genera el bias menos para la primer capa (input layer)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # asigna los pesos de una capa a la otra para cada par de capas

    # en cada paso tomando el input calcula la func activación y luego eso se tranforma en el nuevo input para la sig capa
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) # we apply a single step of gradient descent, which updates the network weights and biases according to a single iteration
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # nabla_b y nabla_w son acumuladores (inicialmente ceros). Guardarán la suma de gradientes de biases y pesos de todos los ejemplos del mini_batch.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # delta_nabla_b: cuánto debe ajustarse cada bias y delta_nabla_w: cuánto debe ajustarse cada peso.
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # calcula los gradientes de la función de costo respecto a cada peso y bias de la red. Osea calcula el error que cada peso/bias cometió en un ejemplo
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(self.feedforward(x) > 0.5, vectorized_result_binary(y)) for (x, y) in test_data]
        # Convertimos a booleanos 0/1 comparando con 0.5
        return sum(int(np.array_equal(pred, actual)) for (pred, actual) in test_results)


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



"""
>>> import mnist_loader_b
>>> training_data, validation_data, test_data = mnist_loader_b.load_data_wrapper()
>>> import network_b
>>> net = network_b.Network([784, 30, 4])
>>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
Epoch 0: 8212 / 10000
Epoch 1: 8706 / 10000
Epoch 2: 8948 / 10000
Epoch 3: 8952 / 10000
Epoch 4: 9045 / 10000
Epoch 5: 8978 / 10000
Epoch 6: 9057 / 10000
Epoch 7: 9100 / 10000
Epoch 8: 9127 / 10000
Epoch 9: 9069 / 10000
Epoch 10: 9167 / 10000
Epoch 11: 9086 / 10000
Epoch 12: 9140 / 10000
Epoch 13: 9169 / 10000
Epoch 14: 9138 / 10000
Epoch 15: 9140 / 10000
Epoch 16: 9154 / 10000
Epoch 17: 9174 / 10000
Epoch 18: 9171 / 10000
Epoch 19: 9188 / 10000
Epoch 20: 9165 / 10000
Epoch 21: 9211 / 10000
Epoch 22: 9167 / 10000
Epoch 23: 9137 / 10000
Epoch 24: 9194 / 10000
Epoch 25: 9148 / 10000
Epoch 26: 9171 / 10000
Epoch 27: 9212 / 10000
Epoch 28: 9160 / 10000
Epoch 29: 9169 / 10000



No es tan buena porque (citado del tutorial/libro):
Supposing the neural network functions in this way, we can give a plausible explanation for why it's better to have 10
outputs from the network, rather than 4. If we had 4
outputs, then the first output neuron would be trying to decide what the most significant bit of the digit was. And there's no easy way to relate that most significant
bit to simple shapes like those shown above. It's hard to imagine that there's any good historical reason the component shapes of the digit will be closely related to (say)
the most significant bit in the output.
"""
