import csv


def print_results(v_net_opt, output_by_iteration, filename, method):
    # count connections for best network
    v_net_error, count_hidden_neurons, count_connections = method.get_iteration_data(v_net_opt)
    print("v_net_opt error on validation set: {}".format(v_net_error))
    print("v_net_opt n_hidden: {}".format(count_hidden_neurons))
    print("v_net_opt n_connect: {}".format(count_connections))
    print("v_net_opt hidden: {}".format(v_net_opt['particle']['hidden']))
    print("v_net_opt w_input: {}".format(v_net_opt['particle']['w_input']))
    print("v_net_opt w_hidden: {}".format(v_net_opt['particle']['w_hidden']))

    if all(name in ['b_input', 'b_hidden'] for name in v_net_opt['particle']):
        print("v_net_opt b_input: {}".format(v_net_opt['particle']['b_input']))
        print("v_net_opt b_hidden: {}".format(v_net_opt['particle']['b_hidden']))
    else:
        print("v_net_opt bias included on weight matrix")

    if 'c_input' in v_net_opt['particle']:
        print("v_net_opt c_input: {}".format(v_net_opt['particle']['c_input']))
        print("v_net_opt c_hidden: {}".format(v_net_opt['particle']['c_hidden']))

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for iteration, triple in output_by_iteration:
            triple = list(triple)
            writer.writerow([iteration, triple[0], triple[1], triple[2]])


def run_in_test_set(v_net_opt, X_test, y_test, method):
    test = method.forward_propagate(v_net_opt, X_test, y_test)
    accuracy = (1 - test['particle']['error']) * 100
    print("v_net_opt error on test set: {}".format(test['particle']['error']))
    print("v_net_opt accuracy on test set: {}%".format(accuracy))