import copy
import Datasets
from random import uniform
from random import random
from math import exp
from sklearn.model_selection import train_test_split


def initialize_population(n_particles, n_hidden, n_output):
    n_input = len(X_train[0])
    population = []
    for i in range(n_particles):

        hidden_neurons = [0 for i in range(n_hidden)]
        w_input = [[uniform(-1, 1) for i in range(n_input + 1)] for j in range(n_hidden)]
        w_hidden = [[uniform(-1, 1) for i in range(n_hidden + 1)] for j in range(n_output)]
        v_w_input = [[uniform(-1, 1) for i in range(n_input + 1)] for j in range(n_hidden)]
        v_w_hidden = [[uniform(-1, 1) for i in range(n_hidden + 1)] for j in range(n_output)]

        particle = {'particle': {'hidden_neurons': hidden_neurons, 'error': float('inf'), 'n_output': n_output,
                                 'w_input': w_input, 'w_hidden': w_hidden,
                                 'v_w_input': v_w_input, 'v_w_hidden': v_w_hidden},
                    'p_best': {'error': float('inf')}}

        population.append(particle)

    return population


# Calcula a ativação do neurônio para a entrada
def activate(weights, inputs):

    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]

    return activation


# Transfere a ativação do neurônio
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, input_data, expected_outputs):

    H = copy.deepcopy(network["particle"]["hidden_neurons"])
    n_output = copy.deepcopy(network["particle"]["n_output"])
    w_input = copy.deepcopy(network["particle"]["w_input"])
    w_hidden = copy.deepcopy(network["particle"]["w_hidden"])

    error = 0
    for t, example in enumerate(input_data):
        # input layer -> hidden layer
        for i in range(len(H)):
            activation = activate(w_input[i], example)
            H[i] = transfer(activation)

        # hidden layer -> output layer
        output = []
        for i in range(n_output):
            activation = activate(w_hidden[i], H)
            output.append(transfer(activation))

        prediction = output.index(max(output))

        #print("pred: {}, expect: {}".format(prediction, expected_outputs[t]))

        if prediction != expected_outputs[t]:
            error += 1

    classification_error = error / len(input_data)
    #print("ERROR: %s" % (classification_error))

    network['particle'].update({'error': classification_error})

    if network['particle']['error'] < network['p_best']['error']:
        network.update({'p_best': copy.deepcopy(network['particle'])})
        # print("p_best: {}".format(network['p_best']['error']))

    return network


def update_velocity(particle, g_best, v_lim):
    c1, c2 = 2.55, 2.55
    inertia_weight = 0.8

    v_w_input = copy.deepcopy(particle['particle']['v_w_input'])
    x_w_inp = particle['particle']['w_input']
    p_best_w_inp = particle['p_best']['w_input']
    g_best_w_inp = g_best['particle']['w_input']

    for i in range(len(v_w_input)):
        for j in range(len(v_w_input[i])):
            v_w_input[i][j] = inertia_weight * v_w_input[i][j] + (
                    c1 * random() * (p_best_w_inp[i][j] - x_w_inp[i][j])) + (
                    c2 * random() * (g_best_w_inp[i][j] - x_w_inp[i][j]))
            if v_w_input[i][j] > v_lim[1]:
                v_w_input[i][j] = v_lim[1]
            elif v_w_input[i][j] < v_lim[0]:
                v_w_input[i][j] = v_lim[0]

    v_w_hidden = copy.deepcopy(particle['particle']['v_w_hidden'])
    x_w_hid = particle['particle']['w_hidden']
    p_best_w_hid = particle['p_best']['w_hidden']
    g_best_w_hid = g_best['particle']['w_hidden']

    for i in range(len(v_w_hidden)):
        for j in range(len(v_w_hidden[i])):
            v_w_hidden[i][j] = inertia_weight * v_w_hidden[i][j] + (
                    c1 * random() * (p_best_w_hid[i][j] - x_w_hid[i][j])) + (
                    c2 * random() * (g_best_w_hid[i][j] - x_w_hid[i][j]))
            if v_w_hidden[i][j] > v_lim[1]:
                v_w_hidden[i][j] = v_lim[1]
            elif v_w_hidden[i][j] < v_lim[0]:
                v_w_hidden[i][j] = v_lim[0]

    particle['particle']['v_w_input'] = copy.deepcopy(v_w_input)
    particle['particle']['v_w_hidden'] = copy.deepcopy(v_w_hidden)
    return copy.deepcopy(particle)


def update_position(particle, p_lim):
    v_w_input = copy.deepcopy(particle['particle']['v_w_input'])
    v_w_hidden = copy.deepcopy(particle['particle']['v_w_hidden'])
    w_input = copy.deepcopy(particle['particle']['w_input'])
    w_hidden = copy.deepcopy(particle['particle']['w_hidden'])

    for i in range(len(w_input)):
        for j in range(len(w_input[i])):
            w_input[i][j] = w_input[i][j] + v_w_input[i][j]
            if w_input[i][j] > p_lim[1]:
                w_input[i][j] = p_lim[1]
            elif w_input[i][j] < p_lim[0]:
                w_input[i][j] = p_lim[0]

    for i in range(len(w_hidden)):
        for j in range(len(w_hidden[i])):
            w_hidden[i][j] = w_hidden[i][j] + v_w_hidden[i][j]
            if w_hidden[i][j] > p_lim[1]:
                w_hidden[i][j] = p_lim[1]
            elif w_hidden[i][j] < p_lim[0]:
                w_hidden[i][j] = p_lim[0]

    particle['particle']['w_input'] = copy.deepcopy(w_input)
    particle['particle']['w_hidden'] = copy.deepcopy(w_hidden)
    return copy.deepcopy(particle)


dataset = Datasets.load_seeds()

classes = set([example[-1] for example in dataset])

print("training examples: {}".format(len(dataset)))
print("classes: {}".format(classes))

x = [example[0:len(example)-1] for example in dataset]
y = [example[-1] for example in dataset]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

population = initialize_population(30, 3, len(classes))

g_best = {'particle': {"error": float('inf')}}
i = 0

while i < 1000:
    for p in range(len(population)):
        particle = forward_propagate(population[p], X_train, y_train)

        if particle['particle']["error"] < g_best['particle']["error"]:
            g_best.update(copy.deepcopy(particle))

        particle = update_velocity(particle, g_best, v_lim=[-1,1])
        population[p] = update_position(particle, p_lim=[-1, 1])

    print("i: {}, g_best: {}".format(i, g_best['particle']['error']))
    i += 1