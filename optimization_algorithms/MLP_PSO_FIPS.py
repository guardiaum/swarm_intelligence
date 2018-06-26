import copy
from random import uniform
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

        if prediction != expected_outputs[t]:
            error += 1

    classification_error = error / len(input_data)

    network['particle'].update({'hidden': H})
    network['particle'].update({'error': classification_error})

    if network['particle']['error'] < network['p_best']['error']:
        network.update({'p_best': copy.deepcopy(network['particle'])})
        # print("p_best: {}".format(network['p_best']['error']))

    return network


def get_iteration_data(g_best):
    count_hidden_neurons = 0
    count_connections = 'ALL'

    hidden = g_best['particle']['hidden']

    for i in range(len(hidden)):
        if hidden[i] > 0:
            count_hidden_neurons += 1

    return g_best['particle']['error'], count_hidden_neurons, count_connections


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


def update_velocity(inertia_w, w_type, particle, neighbors):
    phi = calculate_phi(len(neighbors))
    p_input, p_hidden = calculate_p(phi, w_type, particle, neighbors)

    phi_ = 0
    for i in range(0, len(phi)):
        phi_ += phi[i]

    v_input = copy.deepcopy(particle['particle']['v_w_input'])
    v_hidden = copy.deepcopy(particle['particle']['v_w_hidden'])

    for i in range(len(particle['particle']['w_input'])):
        for j in range(len(particle['particle']['w_input'][0])):
            v_input[i][j] = inertia_w * (v_input[i][j] + phi_ * (p_input[i][j] - particle['particle']['w_input'][i][j]))

    for i in range(len(particle['particle']['w_hidden'])):
        for j in range(len(particle['particle']['w_hidden'][0])):
            v_hidden[i][j] = inertia_w * (v_hidden[i][j] + phi_ * (p_hidden[i][j] - particle['particle']['w_hidden'][i][j]))

    particle['particle']['v_w_input'] = copy.deepcopy(v_input)
    particle['particle']['v_w_hidden'] = copy.deepcopy(v_hidden)

    return particle


def calculate_phi(n_size):
    phi_limit = 4.1 / n_size
    phi = []
    x = uniform(0, phi_limit)
    for i in range(0, n_size):
        phi.append(x)
    return phi


def calculate_p(phi, w_type, particle, neighbors):
    p_input = []
    p_hidden = []

    for i in range(len(particle['particle']['w_input'])):
        aux = []

        for j in range(len(particle['particle']['w_input'][0])):
            numerator= 0
            divisor= 0

            for k in range(0, len(neighbors)):
                w_value = get_w(copy.deepcopy(particle['particle']['w_input'][i]),
                                copy.deepcopy(neighbors[k]['p_best']['w_input'][i]),
                                copy.deepcopy(neighbors[k]['p_best']['error']), w_type)
                numerator += w_value * phi[k] * neighbors[k]['p_best']['w_input'][i][j]
                divisor += w_value * phi[k]

            aux.append(numerator / divisor)
        p_input.append(aux)

    for i in range(len(particle['particle']['w_hidden'])):
        aux = []

        for j in range(len(particle['particle']['w_hidden'][0])):
            numerator = 0
            divisor = 0

            for k in range(0, len(neighbors)):
                w_value = get_w(copy.deepcopy(particle['particle']['w_hidden'][i]),
                                copy.deepcopy(neighbors[k]['p_best']['w_hidden'][i]),
                                copy.deepcopy(neighbors[k]['p_best']['error']), w_type)
                numerator += w_value * phi[k] * neighbors[k]['p_best']['w_hidden'][i][j]
                divisor += w_value * phi[k]

            aux.append(numerator/ divisor)
        p_hidden.append(aux)

    return p_input, p_hidden


def get_w(particle_w, neighbor_w, neighbor_pbest_error, w_type):
    if w_type == 'static':  # FIPS
        return 0.5
    elif w_type == 'fitness':  # wFIPS
        return neighbor_pbest_error
    elif w_type == 'distance':  # wdFIPS
        distance1 = Neighborhood.euclidian_dist(particle_w, neighbor_w)
        if distance1 < 0.001:
            distance1 = 0.001
        return distance1


# generaliziation loss used to stop execution when reach criteria
def generalization_loss(v_net_opt, v_net_current):
    if v_net_opt['particle']['error'] == 0:
        return 0
    else: return 100 * (v_net_current['particle']['error'] / v_net_opt['particle']['error']) - 1


''' weight_method: default, fitness e distance'''
def run(X_train, X_val, y_train, y_val,
        n_particles, n_hidden, n_output,
        max_iter, check_gloss, neighborhood_size, weight_method,
        inertia_weight, p_lim):

    population = initialize_population(n_particles, len(X_train[0]), n_hidden, n_output)

    g_best = {'particle': {"error": float('inf')}}
    g_loss = 0
    stagnation_count = 0
    v_net_opt = None
    hist = []
    output_by_iteration = []
    iteration = 0

    for p in population:
        p = forward_propagate(p, X_train, y_train)

    while iteration < max_iter and g_loss < 5 and stagnation_count < 3:

        for i, p in enumerate(population):
            neighbors = Neighborhood.get_neighbors(i, population, neighborhood_size)

            particle = update_velocity(inertia_weight, weight_method, p, neighbors)

            particle = update_position(particle, p_lim)

            population[i] = forward_propagate(particle, X_train, y_train)

            if particle['particle']["error"] < g_best['particle']["error"]:
                g_best.update(copy.deepcopy(particle))

        print("i: {}, g_best: {}".format(iteration, g_best['particle']['error']))

        output_by_iteration.append([iteration, get_iteration_data(g_best)])

        hist.append(g_best)

        # get error in validation ser for v_net_current and v_net_opt
        if (iteration > check_gloss) and (iteration % 100 == 0):

            v_net_current = forward_propagate(g_best, X_val, y_val)
            min_v_net_from_hist = min(hist, key=lambda x: x['particle']['error'])
            v_net_opt = forward_propagate(min_v_net_from_hist, X_val, y_val)

            if v_net_current['particle']["error"] < v_net_opt['particle']["error"]:
                v_net_opt = v_net_current
                stagnation_count = 0
            else:
                g_loss = generalization_loss(v_net_opt, v_net_current)
                stagnation_count = stagnation_count + 1

        iteration += 1

    return v_net_opt, output_by_iteration