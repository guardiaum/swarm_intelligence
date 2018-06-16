import MLP

# Algoritmo de backpropagation com Stochastic Gradient Descent
def backpropagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1  # input neurons
    n_outputs = len(set([row[-1] for row in train]))  # output neurons

    # print("INPUT NEURONS: {}; OUTPUT NEURONS: {}".format(n_inputs, n_outputs))

    network = MLP.initialize_network(n_inputs, n_hidden, n_outputs)

    train_network(network, train, l_rate, n_epoch, n_outputs)

    predictions = list()
    for row in test:
        prediction = MLP.predict(network, row)
        predictions.append(prediction)

    return predictions

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
            neuron['delta'] = errors[j] * MLP.transfer_derivative(neuron['output'])


# Treinamento da rede para um número fixo de períodos
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = MLP.forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)


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