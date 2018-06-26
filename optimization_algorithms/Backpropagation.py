from random import random
from math import exp


# Inicializa a rede
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Propaga a entrada até a saída da rede
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calcula a ativação do neurônio para a entrada
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


# Transfere a ativação do neurônio
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Calcula a derivada da saída de um neurônio
def transfer_derivative(output):
    return output * (1.0 - output)


# classifica o exemplo
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Algoritmo de backpropagation com Stochastic Gradient Descent
def backpropagation(train, l_rate, n_epoch, n_hidden, n_outputs):
    n_inputs = len(train[0]) - 1  # input neurons
    #n_outputs = len(set([row[-1] for row in train]))  # output neurons

    network = initialize_network(n_inputs, n_hidden, n_outputs)

    output_by_iteration = train_network(network, train, l_rate, n_epoch, n_hidden, n_outputs)

    return network, output_by_iteration

# Propaga o erro de volta (da saída até à camada de entrada)
# armazena o erro nos neurônios de cada camada
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:  # camada escondida
            for j in range(len(layer)):
                error = 0.0
                # cada neurônio da camada escondida
                # ponderada o seu peso com o erro
                # advindo da camada de saída
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:  # camada de saída
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Treinamento da rede para um número fixo de períodos
def train_network(network, train, l_rate, n_epoch, n_hidden, n_outputs):
    output_by_iteration = []
    for epoch in range(n_epoch):
        error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1

            if outputs.index(max(outputs)) != expected.index(1):
                error += 1

            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

        error = error / len(train)

        print("i: {}, error: {}".format(epoch, error))

        output_by_iteration.append([epoch, error, n_hidden, 'ALL'])

    return output_by_iteration


# Atualiza os pesos da rede de acordo com o erro
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]

        if i != 0:  # caso não seja a camada de entrada
            inputs = [neuron['output'] for neuron in network[i - 1]]

        for neuron in network[i]:
            for j in range(len(inputs)):  # atualiza os pesos do neuronio
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            # atualiza o bias
            neuron['weights'][-1] += l_rate * neuron['delta']