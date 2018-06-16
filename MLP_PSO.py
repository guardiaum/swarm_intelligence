from random import uniform
from math import exp
from math import sqrt
from random import random


# Transfere a ativação do neurônio
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


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

    if network['p_best'] is None or network['error'] < network['p_best']['error']:
        network['p_best'] = network

    # print("network['error']: {}".format(network["error"]))

    return network


def initialize_population(n_particles, n_input, n_hidden, n_output):

    population = [[]] * 50
    print(len(population))
    for particle in range(n_particles):
        H = [None for i in range(n_hidden)]

        B_input = [uniform(-1, 1) for i in range(n_hidden)]
        B_hidden = [uniform(-1, 1) for i in range(n_output)]

        C_input = [[uniform(-1, 1) for y in range(n_input)] for x in range(n_hidden)]
        C_hidden = [[uniform(-1, 1) for y in range(n_hidden)] for x in range(n_output)]

        W_input = [[uniform(-1, 1) for y in range(n_input)] for x in range(n_hidden)]
        W_hidden = [[uniform(-1, 1) for y in range(n_hidden)] for x in range(n_output)]

        v_b_input = [uniform(-1, 1) for i in range(n_input)]
        v_b_hidden = [uniform(-1, 1) for i in range(n_output)]
        v_c_input = [[uniform(-1, 1) for y in range(n_input)] for x in range(n_hidden)]
        v_c_hidden = [[uniform(-1, 1) for y in range(n_hidden)] for x in range(n_output)]
        v_w_input = [[uniform(-1, 1) for y in range(n_input)] for x in range(n_hidden)]
        v_w_hidden = [[uniform(-1, 1) for y in range(n_hidden)] for x in range(n_output)]

        population[particle] = {
            "p_best": None, "error": None,
            "hidden": H, "n_output": n_output,
            "v_b_input": v_b_input, "v_b_hidden": v_b_hidden,
            "v_c_input": v_c_input, "v_c_hidden": v_c_hidden,
            "v_w_input": v_w_input, "v_w_hidden": v_w_hidden,
            "b_input": B_input, "b_hidden": B_hidden,
            "c_input": C_input, "c_hidden": C_hidden,
            "w_input": W_input, "w_hidden": W_hidden}

    return population


def run(input_data, n_particles, n_hidden, n_output, max_iter):
    training_examples = [example[0:len(example)-1] for example in input_data]
    expected_outputs = [example[-1] for example in input_data]

    # input neurons
    n_input = len(training_examples[0])

    population = initialize_population(n_particles, n_input, n_hidden, n_output)

    i = 0
    g_loss = 0
    stagnation_count = 0
    g_best = {"error": float('inf')}
    v_net_current = None
    v_net_opt = None
    hist = []

    while i < max_iter and g_loss < 5 and stagnation_count < 3:
        print("iteration: {}".format(i))
        # computes particles local best (pBest) with lower training error
        for p in range(len(population)):
            # pass particle/network to propagate input_data 'till obtain training error
            particle = forward_propagate(population[p], training_examples, expected_outputs)

            # chooses the swarm best gBest with lower training error
            if particle["error"] < g_best["error"]:
                g_best = particle
                hist.append(g_best)

            particle = update_velocities(particle, g_best)
            population[p] = update_positions(particle)

        v_net_current = g_best
        v_net_opt = min(hist, key=lambda x: x['error'])

        if (i > 500) and (i % 100 == 0):
            if v_net_current["error"] < v_net_opt["error"]:
                v_net_opt = v_net_current
                stagnation_count = 0
            else:
                g_loss = generalization_loss(v_net_opt, v_net_current)
                stagnation_count = stagnation_count + 1
                hist = []
        print("gBest_error: {}".format(v_net_opt['error']))
        i += 1

    return v_net_opt


def generalization_loss(v_net_opt, v_net_current):
    return 100 * ((float(v_net_current['error']) / float(v_net_opt['error'])) - 1)


def update_velocities(particle, g_best):
    v_b_input = particle["v_b_input"]
    v_b_hidden = particle["v_b_hidden"]
    v_c_input = particle["v_c_input"]
    v_c_hidden = particle["v_c_hidden"]
    v_w_input = particle["v_w_input"]
    v_w_hidden = particle["v_w_hidden"]

    r1, r2 = random(), random()

    particle['v_b_input'], particle['v_b_hidden'] = update_bias_velocity(g_best, particle, v_b_hidden, v_b_input, r1, r2)
    particle['v_c_input'], particle['v_c_hidden'] = update_connections_velocity(g_best, particle, v_c_hidden, v_c_input, r1, r2)
    particle['v_w_input'], particle['v_w_hidden'] = update_weights_velocity(g_best, particle, v_w_hidden, v_w_input, r1, r2)

    return particle


def update_weights_velocity(g_best, particle, v_w_hidden, v_w_input, r1, r2):
    phi = 4.1
    c1 = 2.55
    c2 = 1.55
    chi = 2 / (2 - phi - sqrt(phi ** 2 - 4 * phi))

    for j in range(len(v_w_input)):  # input connections to neurons of hidden layer
        for i in range(len(v_w_input[j])):  # each input connections to hidden neuron
            v_w_input[j][i] = chi * (v_w_input[j][i] +
                                  (c1 * r1 * (particle["p_best"]["w_input"][j][i] - particle["w_input"][j][i])) +
                                  (c2 * r2 * (g_best["w_input"][j][i] - particle["w_input"][j][i])))
            if 1 < v_w_input[j][i]:
                v_w_input[j][i] = 1
            elif v_w_input[j][i] < -1:
                v_w_input[j][i] = -1

    for j in range(len(v_w_hidden)):  # hidden connections to neurons of output layer
        for i in range(len(v_w_hidden[j])):  # each hidden connections to output neuron
            v_w_hidden[j][i] = chi * (v_w_hidden[j][i] +
                                   (c1 * r1 * (float(particle["p_best"]["w_hidden"][j][i]) - float(particle["w_hidden"][j][i]))) +
                                   (c2 * r2 * (g_best["w_hidden"][j][i] - particle["w_hidden"][j][i])))
            if 1 < v_w_hidden[j][i]:
                v_w_hidden[j][i] = 1
            elif v_w_hidden[j][i] < -1:
                v_w_hidden[j][i] = -1

    return v_w_input, v_w_hidden


def update_connections_velocity(g_best, particle, v_c_hidden, v_c_input, r1, r2):
    phi = 4.1
    c1 = 2.55
    c2 = 1.55
    chi = 2 / (2 - phi - sqrt(phi ** 2 - 4 * phi))

    for j in range(len(v_c_input)):  # input connections to neurons of hidden layer
        for i in range(len(v_c_input[j])):  # each input connections to hidden neuron
            v_c_input[j][i] = chi * (v_c_input[j][i] +
                                  (c1 * r1 * (particle["p_best"]["c_input"][j][i] - particle["c_input"][j][i])) +
                                  (c2 * r2 * (g_best["c_input"][j][i] - particle["c_input"][j][i])))
            if 1 < v_c_input[j][i]:
                v_c_input[j][i] = 1
            elif v_c_input[j][i] < -1:
                v_c_input[j][i] = -1

    for j in range(len(v_c_hidden)):  # hidden connections to neurons of output layer
        for i in range(len(v_c_hidden[j])):  # each hidden connections to output neuron
            v_c_hidden[j][i] = chi * (v_c_hidden[j][i] +
                                   (c1 * r1 * (particle["p_best"]["c_hidden"][j][i] - particle["c_hidden"][j][i])) +
                                   (c2 * r2 * (g_best["c_hidden"][j][i] - particle["c_hidden"][j][i])))
            if 1 < v_c_hidden[j][i]:
                v_c_hidden[j][i] = 1
            elif v_c_hidden[j][i] < -1:
                v_c_hidden[j][i] = -1

    return v_c_input, v_c_hidden


def update_bias_velocity(g_best, particle, v_b_hidden, v_b_input, r1, r2):
    phi = 4.1
    c1 = 2.55
    c2 = 1.55
    chi = 2 / (2 - phi - sqrt(phi ** 2 - 4 * phi))

    for i in range(len(v_b_input)):
        v_b_input[i] = chi * (v_b_input[i] +
                              (c1 * r1 * (particle["p_best"]["b_input"][i] - particle["b_input"][i])) +
                              (c2 * r2 * (g_best["b_input"][i] - particle["b_input"][i])))
        if 1 < v_b_input[i]:
            v_b_input[i] = 1
        elif v_b_input[i] < -1:
            v_b_input[i] = -1

    for i in range(len(v_b_hidden)):
        v_b_hidden[i] = chi * (v_b_hidden[i] +
                               (c1 * r1 * (particle["p_best"]["b_hidden"][i] - particle["b_hidden"][i])) +
                               (c2 * r2 * (g_best["b_hidden"][i] - particle["b_hidden"][i])))
        if 1 < v_b_hidden[i]:
            v_b_hidden[i] = 1
        elif v_b_hidden[i] < -1:
            v_b_hidden[i] = -1

    return v_b_input, v_b_hidden


def update_positions(particle):
    v_b_input = particle["v_b_input"]
    v_b_hidden = particle["v_b_hidden"]
    v_c_input = particle["v_c_input"]
    v_c_hidden = particle["v_c_hidden"]
    v_w_input = particle["v_w_input"]
    v_w_hidden = particle["v_w_hidden"]

    particle['b_input'], particle['b_hidden'] = update_bias(particle, v_b_input, v_b_hidden)
    particle['c_input'], particle['c_hidden'] = update_connections(particle, v_c_input, v_c_hidden)
    particle['w_input'], particle['w_hidden'] = update_weights(particle, v_w_input, v_w_hidden)
    return particle


def update_bias(particle, v_b_input, v_b_hidden):
    b_input = particle['b_input']

    for i in range(len(b_input)):
        b_input[i] = b_input[i] + v_b_input[i]

    b_hidden = particle['b_hidden']

    for i in range(len(b_hidden)):
        b_hidden[i] = b_hidden[i] + v_b_hidden[i]

    return b_input, b_hidden


def update_connections(particle, v_c_input, v_c_hidden):
    c_input = particle['c_input']
    c_hidden = particle['c_hidden']

    for j in range(len(c_input)):  # input connections to neurons of hidden layer
        for i in range(len(c_input[j])):  # each input connections to hidden neuron
            c_input[j][i] = c_input[j][i] + v_c_input[j][i]

    for j in range(len(c_hidden)):  # input connections to neurons of hidden layer
        for i in range(len(c_hidden[j])):  # each input connections to hidden neuron
            c_hidden[j][i] = c_hidden[j][i] + v_c_hidden[j][i]

    return c_input, c_hidden


def update_weights(particle, v_w_input, v_w_hidden):
    w_input = particle['w_input']
    w_hidden = particle['w_hidden']

    for j in range(len(w_input)):  # input connections to neurons of hidden layer
        for i in range(len(w_input[j])):  # each input connections to hidden neuron
            w_input[j][i] = w_input[j][i] + v_w_input[j][i]

    for j in range(len(w_hidden)):  # input connections to neurons of hidden layer
        for i in range(len(w_hidden[j])):  # each input connections to hidden neuron
            w_hidden[j][i] = w_hidden[j][i] + v_w_hidden[j][i]

    return w_input, w_hidden