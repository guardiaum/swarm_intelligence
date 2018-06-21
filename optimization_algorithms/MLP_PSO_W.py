from random import uniform
from math import exp
from math import sqrt
from random import random
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

    H = copy.deepcopy(network["particle"]["hidden"])
    n_output = copy.deepcopy(network["particle"]["n_output"])
    b_input = copy.deepcopy(network["particle"]["b_input"])
    b_hidden = copy.deepcopy(network["particle"]["b_hidden"])
    w_input = copy.deepcopy(network["particle"]["w_input"])
    w_hidden = copy.deepcopy(network["particle"]["w_hidden"])

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

    network['particle'].update({'hidden': H})
    network['particle'].update({'error': errors / len(input_data)})

    if network['particle']['error'] < network['p_best']['error']:
        network.update({'p_best': copy.deepcopy(network['particle'])})
        print("p_best: {}".format(network['p_best']['error']))

    return network


# initialize particles population
def initialize_population(n_particles, n_input, n_hidden, n_output):

    population = [[]] * n_particles
    print("population: {}".format(len(population)))

    for particle in range(n_particles):
        # initialize hidden layer
        H = [0 for i in range(n_hidden)]

        # initialize biases vectors
        B_input = [uniform(-1, 1) for i in range(n_hidden)]
        B_hidden = [uniform(-1, 1) for i in range(n_output)]

        # initialize weight matrices
        W_input = [[uniform(-1, 1) for y in range(n_input)] for x in range(n_hidden)]
        W_hidden = [[uniform(-1, 1) for y in range(n_hidden)] for x in range(n_output)]

        # initialize velocities
        v_b_input = [uniform(-1, 1) for i in range(n_hidden)]
        v_b_hidden = [uniform(-1, 1) for i in range(n_output)]
        v_w_input = [[uniform(-1, 1) for y in range(n_input)] for x in range(n_hidden)]
        v_w_hidden = [[uniform(-1, 1) for y in range(n_hidden)] for x in range(n_output)]

        population[particle] = {
            'particle': {'error': None,
                         'hidden': H, "n_output": n_output,
                         'v_b_input': v_b_input, 'v_b_hidden': v_b_hidden,
                         'v_w_input': v_w_input, 'v_w_hidden': v_w_hidden,
                         'b_input': B_input, 'b_hidden': B_hidden,
                         'w_input': W_input, 'w_hidden': W_hidden
            },
            'p_best': {'error': float('inf')}}

    return population


def run(X_train, X_val, y_train, y_val, n_particles, n_hidden, n_output, max_iter, v_lim, p_lim):
    # input neurons
    n_input = len(X_train[0])

    population = initialize_population(n_particles, n_input, n_hidden, n_output)

    i = 0
    stagnation_count = 0
    g_loss = 0
    g_best = {'particle': {'error': float('inf')}}
    v_net_opt = None
    hist = []
    output_by_iteration = []

    while i < max_iter and g_loss < 5 and stagnation_count < 3:
        print("iteration: {}".format(i))
        # computes particles local best (pBest) with lower training error
        for p in range(len(population)):
            # pass particle/network to propagate input_data 'till obtain training error
            particle = forward_propagate(population[p], X_train, y_train)

            # chooses the swarm best gBest with lower training error
            if particle['particle']["error"] < g_best['particle']["error"]:
                g_best.update(copy.deepcopy(particle))

            c1 = apply_social_coef_reducing(i, max_iter, c_i=2.55, c_f=1.55)
            c2 = apply_cogn_coef_reducing(i, max_iter, c_i=1.55, c_f=2.55)

            particle = update_velocities(particle, g_best, c1, c2, v_lim)
            population[p] = update_positions(particle, p_lim)

        print("gBest_error: {}".format(g_best['particle']['error']))

        output_by_iteration.append([i, get_iteration_data(g_best)])

        hist.append(g_best)

        # get error in validation ser for v_net_current and v_net_opt
        if (i > 300) and (i % 100 == 0):

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


def apply_social_coef_reducing(i, max_iter, c_i=2.55, c_f=1.55):
    c1 = (c_f - c_i) * (i/max_iter) + c_i
    return c1


def apply_cogn_coef_reducing(i, max_iter, c_i=1.55, c_f=2.55):
    c2 = (c_f - c_i) * (i/max_iter) + c_i
    return c2


# update velocities for each ANN aspect beeing optimized
def update_velocities(particle, g_best, c1, c2, v_lim):
    v_b_input = copy.deepcopy(particle['particle']["v_b_input"])
    v_b_hidden = copy.deepcopy(particle['particle']["v_b_hidden"])
    v_w_input = copy.deepcopy(particle['particle']["v_w_input"])
    v_w_hidden = copy.deepcopy(particle['particle']["v_w_hidden"])

    r1, r2 = random(), random()

    particle['particle']['v_b_input'], particle['particle']['v_b_hidden'] = \
        update_bias_velocity(g_best, particle, v_b_hidden, v_b_input, r1, r2, c1, c2, v_lim)

    particle['particle']['v_w_input'], particle['particle']['v_w_hidden'] = \
        update_weights_velocity(g_best, particle, v_w_hidden, v_w_input, r1, r2, c1, c2, v_lim)

    return copy.deepcopy(particle)


def update_weights_velocity(g_best, particle, v_w_hidden, v_w_input, r1, r2, c1, c2, v_lim):
    phi = c1 + c2
    chi = 2 / (2 - phi - sqrt(phi ** 2 - 4 * phi))

    for j in range(len(v_w_input)):  # input connections to neurons of hidden layer
        for i in range(len(v_w_input[j])):  # each input connections to hidden neuron
            v_w_input[j][i] = chi * (v_w_input[j][i] +
                                  (c1 * r1 * (particle['p_best']['w_input'][j][i] - particle['particle']['w_input'][j][i])) +
                                  (c2 * r2 * (g_best['particle']['w_input'][j][i] - particle['particle']['w_input'][j][i])))
            if v_w_input[j][i] > v_lim[1]:
                v_w_input[j][i] = v_lim[1]
            elif v_w_input[j][i] < v_lim[0]:
                v_w_input[j][i] = v_lim[0]

    for j in range(len(v_w_hidden)):  # hidden connections to neurons of output layer
        for i in range(len(v_w_hidden[j])):  # each hidden connections to output neuron
            v_w_hidden[j][i] = chi * (v_w_hidden[j][i] +
                                   (c1 * r1 * (float(particle["p_best"]["w_hidden"][j][i]) - float(particle['particle']["w_hidden"][j][i]))) +
                                   (c2 * r2 * (g_best['particle']["w_hidden"][j][i] - particle['particle']["w_hidden"][j][i])))
            if v_w_hidden[j][i] > v_lim[1]:
                v_w_hidden[j][i] = v_lim[1]
            elif v_w_hidden[j][i] < v_lim[0]:
                v_w_hidden[j][i] = v_lim[0]

    return copy.deepcopy(v_w_input), copy.deepcopy(v_w_hidden)


def update_bias_velocity(g_best, particle, v_b_hidden, v_b_input, r1, r2, c1, c2, v_lim):
    phi = c1 + c2
    chi = 2 / (2 - phi - sqrt(phi ** 2 - 4 * phi))

    for i in range(len(v_b_input)):
        v_b_input[i] = chi * (v_b_input[i] +
                              (c1 * r1 * (particle["p_best"]["b_input"][i] - particle['particle']["b_input"][i])) +
                              (c2 * r2 * (g_best['particle']["b_input"][i] - particle['particle']["b_input"][i])))
        if v_b_input[i] > v_lim[1]:
            v_b_input[i] = v_lim[1]
        elif v_b_input[i] < v_lim[0]:
            v_b_input[i] = v_lim[0]

    for i in range(len(v_b_hidden)):
        v_b_hidden[i] = chi * (v_b_hidden[i] +
                               (c1 * r1 * (particle["p_best"]["b_hidden"][i] - particle['particle']["b_hidden"][i])) +
                               (c2 * r2 * (g_best['particle']["b_hidden"][i] - particle['particle']["b_hidden"][i])))
        if v_b_hidden[i] > v_lim[1]:
            v_b_hidden[i] = v_lim[1]
        elif v_b_hidden[i] < v_lim[0]:
            v_b_hidden[i] = v_lim[0]

    return copy.deepcopy(v_b_input), copy.deepcopy(v_b_hidden)


# update positions for each ANN aspect beeing optimized
def update_positions(particle, p_lim):
    v_b_input = copy.deepcopy(particle['particle']["v_b_input"])
    v_b_hidden = copy.deepcopy(particle['particle']["v_b_hidden"])
    v_w_input = copy.deepcopy(particle['particle']["v_w_input"])
    v_w_hidden = copy.deepcopy(particle['particle']["v_w_hidden"])

    particle['particle']['b_input'], particle['particle']['b_hidden'] = update_bias(particle, v_b_input, v_b_hidden, p_lim)
    particle['particle']['w_input'], particle['particle']['w_hidden'] = update_weights(particle, v_w_input, v_w_hidden, p_lim)
    return copy.deepcopy(particle)


def update_bias(particle, v_b_input, v_b_hidden, p_lim):
    b_input = particle['particle']['b_input']
    b_hidden = particle['particle']['b_hidden']

    for i in range(len(b_input)):
        b_input[i] = b_input[i] + v_b_input[i]
        if b_input[i] > p_lim[1]:
            b_input[i] = p_lim[1]
        elif b_input[i] < p_lim[0]:
            b_input[i] = p_lim[0]

    for i in range(len(b_hidden)):
        b_hidden[i] = b_hidden[i] + v_b_hidden[i]
        if b_hidden[i] > p_lim[1]:
            b_hidden[i] = p_lim[1]
        elif b_hidden[i] < p_lim[0]:
            b_hidden[i] = p_lim[0]

    return copy.deepcopy(b_input), copy.deepcopy(b_hidden)


def update_weights(particle, v_w_input, v_w_hidden, p_lim):
    w_input = particle['particle']['w_input']
    w_hidden = particle['particle']['w_hidden']

    for j in range(len(w_input)):  # input connections to neurons of hidden layer
        for i in range(len(w_input[j])):  # each input connections to hidden neuron
            w_input[j][i] = w_input[j][i] + v_w_input[j][i]
            if w_input[j][i] > p_lim[1]:
                w_input[j][i] = p_lim[1]
            elif w_input[j][i] < p_lim[0]:
                w_input[j][i] = p_lim[0]

    for j in range(len(w_hidden)):  # input connections to neurons of hidden layer
        for i in range(len(w_hidden[j])):  # each input connections to hidden neuron
            w_hidden[j][i] = w_hidden[j][i] + v_w_hidden[j][i]
            if w_hidden[j][i] > p_lim[1]:
                w_hidden[j][i] = p_lim[1]
            elif w_hidden[j][i] < p_lim[0]:
                w_hidden[j][i] = p_lim[0]

    return copy.deepcopy(w_input), copy.deepcopy(w_hidden)