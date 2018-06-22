from util import Datasets
from optimization_algorithms import MLP_CS_W
from sklearn.model_selection import train_test_split
from util import Results
import csv
from beans import function as fn


def print_mlp_cs_results(v_net_opt, output_by_iteration, filename, method):
    # count connections for best network
    net_error, count_hidden_neurons, count_connections = method.get_iteration_data(v_net_opt)
    print("v_net_opt error: {}".format(net_error))
    print("v_net_opt n_hidden: {}".format(count_hidden_neurons))
    print("v_net_opt n_connect: {}".format(count_connections))
    print("v_net_opt hidden: {}".format(v_net_opt['hidden']))
    print("v_net_opt b_input: {}".format(v_net_opt['b_input']))
    print("v_net_opt b_hidden: {}".format(v_net_opt['b_hidden']))
    print("v_net_opt w_input: {}".format(v_net_opt['w_input']))
    print("v_net_opt w_hidden: {}".format(v_net_opt['w_hidden']))

    if 'c_input' in v_net_opt:
        print("v_net_opt c_input: {}".format(v_net_opt['c_input']))
        print("v_net_opt c_hidden: {}".format(v_net_opt['c_hidden']))

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for iteration, triple in output_by_iteration:
            triple = list(triple)
            writer.writerow([iteration, triple[0], triple[1], triple[2]])


def mlp_cs_in_test_set(v_net_opt, X_test, y_test, method):
    test = method.forward_propagate(v_net_opt, X_test, y_test)
    print("v_net_opt error test set: {}".format(test['error']))


dataset = Datasets.load_iris()

classes = set([example[-1] for example in dataset])

print("training examples: {}".format(len(dataset)))
print("classes: {}".format(classes))

x = [example[0:len(example)-1] for example in dataset]
y = [example[-1] for example in dataset]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

v_net_opt, output_by_iteration = MLP_CS_W.run(X_train, X_val, y_train, y_val, n_hidden=5, n_output=len(classes))

print_mlp_cs_results(v_net_opt, output_by_iteration, "results/output_mlp_cs_w_cancer.csv", MLP_CS_W)
mlp_cs_in_test_set(v_net_opt, X_test, y_test, fn)