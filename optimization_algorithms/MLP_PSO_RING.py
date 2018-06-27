import copy
from random import uniform
from random import random
from math import exp
from util.Neighborhood import Neighborhood


def initialize_population(n_particles, n_input, n_hidden, n_output):
    population = []
    for i in range(n_particles):

        hidden_neurons = [0 for i in range(n_hidden)]
        w_input = [[uniform(-1, 1) for i in range(n_input + 1)] for j in range(n_hidden)]
        w_hidden = [[uniform(-1, 1) for i in range(n_hidden + 1)] for j in range(n_output)]
        v_w_input = [[uniform(-1, 1) for i in range(n_input + 1)] for j in range(n_hidden)]
        v_w_hidden = [[uniform(-1, 1) for i in range(n_hidden + 1)] for j in range(n_output)]

        particle = {'particle': {'hidden': hidden_neurons, 'error': float('inf'), 'n_output': n_output,
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

    H = copy.deepcopy(network["particle"]["hidden"])
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

    network['particle'].update({'hidden': H})
    network['particle'].update({'error': classification_error})

    if network['particle']['error'] < network['p_best']['error']:
        network.update({'p_best': copy.deepcopy(network['particle'])})
        # print("p_best: {}".format(network['p_best']['error']))

    return network


def update_velocity(particle, l_best, inertia_weight, c1, c2, v_lim):

    v_w_input = copy.deepcopy(particle['particle']['v_w_input'])
    x_w_inp = particle['particle']['w_input']
    p_best_w_inp = particle['p_best']['w_input']
    l_best_w_inp = l_best['particle']['w_input']

    for i in range(len(v_w_input)):
        for j in range(len(v_w_input[i])):
            v_w_input[i][j] = inertia_weight * v_w_input[i][j] + (
                    c1 * random() * (p_best_w_inp[i][j] - x_w_inp[i][j])) + (
                    c2 * random() * (l_best_w_inp[i][j] - x_w_inp[i][j]))
            if v_w_input[i][j] > v_lim[1]:
                v_w_input[i][j] = v_lim[1]
            elif v_w_input[i][j] < v_lim[0]:
                v_w_input[i][j] = v_lim[0]

    v_w_hidden = copy.deepcopy(particle['particle']['v_w_hidden'])
    x_w_hid = particle['particle']['w_hidden']
    p_best_w_hid = particle['p_best']['w_hidden']
    l_best_w_hid = l_best['particle']['w_hidden']

    for i in range(len(v_w_hidden)):
        for j in range(len(v_w_hidden[i])):
            v_w_hidden[i][j] = inertia_weight * v_w_hidden[i][j] + (
                    c1 * random() * (p_best_w_hid[i][j] - x_w_hid[i][j])) + (
                    c2 * random() * (l_best_w_hid[i][j] - x_w_hid[i][j]))
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


def get_iteration_data(g_best):
    count_hidden_neurons = 0
    count_connections = 'ALL'

    hidden = g_best['particle']['hidden']

    for i in range(len(hidden)):
        if hidden[i] > 0:
            count_hidden_neurons += 1

    return g_best['particle']['error'], count_hidden_neurons, count_connections


# generaliziation loss used to stop execution when reach criteria
def generalization_loss(v_net_opt, v_net_current):
    return 100 * (v_net_current['particle']['error'] / v_net_opt['particle']['error']) - 1


def run(X_train, X_val, y_train, y_val,
        n_particles, n_hidden, n_output,
        max_iter, check_gloss, neighborhood_size, inertia_weight, c1, c2, v_lim, p_lim):

    population = initialize_population(n_particles, len(X_train[0]), n_hidden, n_output)

    g_best = {'particle': {"error": float('inf')}}

    g_loss = 0
    stagnation_count = 0
    v_net_opt = None
    hist = []
    output_by_iteration = []
    i = 0

    while i < max_iter and g_loss < 5 and stagnation_count < 3:

        lbests = [None] * len(population)

        for p in range(len(population)):
            particle = forward_propagate(population[p], X_train, y_train)

            if particle['particle']["error"] < g_best['particle']["error"]:
                g_best.update(copy.deepcopy(particle))

        for p in range(len(population)):
            neighbors = Neighborhood.get_static(p, neighborhood_size,len(population))

            lbest = copy.deepcopy(population[p])
            for n in range(len(neighbors)):
                n_index = neighbors[n]

                if population[n_index]['particle']['error'] < lbest['particle']['error']:
                    lbest = copy.deepcopy(population[n_index])

            lbests[p] = copy.deepcopy(lbest)

        for p in range(len(population)):
            lbest = lbests[p]
            particle = update_velocity(particle, lbest, inertia_weight, c1, c2, v_lim)
            population[p] = update_position(particle, p_lim)

        print("i: {}, g_best: {}".format(i, g_best['particle']['error']))

        output_by_iteration.append([i, get_iteration_data(g_best)])

        hist.append(g_best)

        # get error in validation ser for v_net_current and v_net_opt
        if ((i > check_gloss) and ((i + 1) % 100 == 0)) or i == max_iter - 1:

            v_net_current = forward_propagate(g_best, X_val, y_val)
            min_v_net_from_hist = min(hist, key=lambda x: x['particle']['error'])
            v_net_opt = forward_propagate(min_v_net_from_hist, X_val, y_val)

            if v_net_current['particle']["error"] < v_net_opt['particle']["error"]:
                v_net_opt = v_net_current
                stagnation_count = 0
            else:
                g_loss = generalization_loss(v_net_opt, v_net_current)
                stagnation_count = stagnation_count + 1

        i += 1
    return v_net_opt, output_by_iteration