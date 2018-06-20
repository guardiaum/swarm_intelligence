import numpy as np
from math import exp, sqrt
import copy


def activate(weights, inputs, bias):
    activation = bias
    for i in range(len(weights)):
        activation += weights[i] * inputs[i]
    return activation

# Transfere a ativação do neurônio
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# forward propagate examples to output prediction
def forward_propagate(network, input_data, expected_outputs):

    H = copy.deepcopy(network["hidden"])
    n_output = copy.deepcopy(network["n_output"])
    b_input = copy.deepcopy(network["b_input"])
    b_hidden = copy.deepcopy(network["b_hidden"])
    w_input = copy.deepcopy(network["w_input"])
    w_hidden = copy.deepcopy(network["w_hidden"])

    errors = 0
    for t, example in enumerate(input_data):

        for j in range(len(H)):  # calcular a função de ativação para cada neurônio da camada escondida
            H[j] = activate(w_input[j], example, b_input[j])
            H[j] = transfer(H[j])

        output_layer = [0] * n_output

        for k in range(n_output):  # calcular a função de ativação para cada neurônio da camada de saída
            output_layer[k] = activate(w_hidden[k], H, b_hidden[k])
            output_layer[k] = transfer(output_layer[k])

        prediction = output_layer.index(max(output_layer))

        if expected_outputs[t] != prediction:
            errors += 1

    network.update({'hidden': H})
    network.update({'error': errors / len(input_data)})

    return network

"""# forward propagate examples to output prediction
def forward_propagate(network, input_data, expected_outputs):
    H = network["hidden"]
    n_output = network["n_output"]
    b_input = network["b_input"]
    b_hidden = network["b_hidden"]
    w_input = network["w_input"]
    w_hidden = network["w_hidden"]
    c_input = network["c_input"]
    c_hidden = network["c_hidden"]

    errors = 0
    for t, training_example in enumerate(input_data):

        for j in range(len(H)):  # para cada neurônio da camada escondida
            activation_c_inp = 0.0
            for i in range(len(training_example)):  # para cada atributo do exemplo de treinamento (camada de entrada)
                if c_input[j][i] > 0:  # verifies if connection is activated
                    activation_c_inp += w_input[j][i] * training_example[i] + b_input[j]

            H[j] = transfer(activation_c_inp)

        output_layer = [0] * n_output
        for k in range(n_output):  # para cada neurônio da camada de saída
            activation_c_out = 0.0
            for j in range(len(H)):  # para cada neurônio na camada escondida
                if c_hidden[k][j] > 0:  # verifies if connection is activated
                    activation_c_out += w_hidden[k][j] * H[j] + b_hidden[k]

            output_layer[k] = transfer(activation_c_out)

        prediction = output_layer.index(max(output_layer))
        # print("expected:{}, predicted: {}".format(expected_outputs[t], prediction))

        if expected_outputs[t] != prediction:
            errors += 1

    network["error"] = errors / len(input_data)

    #if network['best_egg'] is None or network['error'] < network['best_egg']['error']:
    #    network['best_egg'] = network

    # print("network['error']: {}".format(network["error"]))

    return network
    """
