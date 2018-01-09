import numpy as np


class neural_network:
    def __init__(self, node_list):
        self.weights = []
        self.biases = []

        prev_num = node_list[0]

        for num_nodes in node_list[1:]:
            self.weights.append(np.random.randn(num_nodes, prev_num)/prev_num)
            self.biases.append(np.random.randn(num_nodes, 1))
            prev_num = num_nodes

    @staticmethod
    def load_from_json(json):
        net = neural_network([])
        for layer in json['layers']:
            net.weights.append(np.array(layer['weights']))
            net.biases.append(np.array(layer['biases']))

        return net

    def save_to_json(self):
        layers = []
        for w, b in zip([x.tolist() for x in self.weights],
                        [x.tolist() for x in self.biases]):
            layers.append({'weights': w, 'biases': b})

        json = {'layers': layers}
        return json

    def predict(self, input_signals):
        output = []
        for signal in input_signals:
            signal = np.array(signal, ndmin=2)
            for w, b in zip(self.weights, self.biases):
                signal = np.tanh(w.dot(signal) + b)

            output.append(signal)

        return output

    def train(self, input_signals, target_signals, learning_rate=0.01):
        error_weights = [np.zeros(w.shape) for w in self.weights]
        error_biases = [np.zeros(b.shape) for b in self.biases]

        for input_signal, target in zip(input_signals, target_signals):
            signal = np.array(input_signal, ndmin=2)
            target = np.array(target, ndmin=2)

            gradients = []
            activations = []

            for w, b in zip(self.weights, self.biases):
                activations.append(signal)
                argument = w.dot(signal) + b
                signal = np.tanh(argument)
                gradients.append(1.0 - np.square(signal))

            backprop_values = zip(reversed(self.weights),
                                  reversed(gradients),
                                  reversed(activations),
                                  reversed(error_weights),
                                  reversed(error_biases))

            # LeCun, Efficient BackProp
            # (http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

            # If the error funciton is least squares, the derivative of the
            # error function is simply predicted_value - target_value
            error_actv = signal - target

            for w, grad, actv, error_w, error_b in backprop_values:
                # Equation (7)
                error_arg = grad * error_actv
                error_b += error_arg

                # Equation (8)
                error_w += np.outer(error_arg, actv)

                # Equation (9)
                error_actv = w.T.dot(error_arg)

        scale = len(input_signals)

        # Equation (10)
        self.weights = [w - learning_rate * delta / scale
                        for w, delta in zip(self.weights, error_weights)]

        self.biases = [b - learning_rate * delta / scale
                       for b, delta in zip(self.biases, error_biases)]
