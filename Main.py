from random import seed
import Datasets
import Evaluate
from optimization_algorithms import Backpropagation
from optimization_algorithms.GlobalSwarm import *
import MLP
import MLP_PSO
from sklearn.model_selection import train_test_split
import MLP_PSO_W
import csv


def print_mlp_pso_results(v_net_opt, output_by_iteration, filename, method):
    # count connections for best network
    v_net_error, count_hidden_neurons, count_connections = method.get_iteration_data(v_net_opt)
    print("v_net_opt error: {}".format(v_net_error))
    print("v_net_opt n_hidden: {}".format(count_hidden_neurons))
    print("v_net_opt n_connect: {}".format(count_connections))
    print("v_net_opt hidden: {}".format(v_net_opt['particle']['hidden']))
    print("v_net_opt b_input: {}".format(v_net_opt['particle']['b_input']))
    print("v_net_opt b_hidden: {}".format(v_net_opt['particle']['b_hidden']))
    print("v_net_opt w_input: {}".format(v_net_opt['particle']['w_input']))
    print("v_net_opt w_hidden: {}".format(v_net_opt['particle']['w_hidden']))

    if 'c_input' in v_net_opt['particle']:
        print("v_net_opt c_input: {}".format(v_net_opt['particle']['c_input']))
        print("v_net_opt c_hidden: {}".format(v_net_opt['particle']['c_hidden']))

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for iteration, triple in output_by_iteration:
            triple = list(triple)
            writer.writerow([iteration, triple[0], triple[1], triple[2]])


def mlp_pso_in_test_set(v_net_opt, X_test, y_test, method):
    test = method.forward_propagate(v_net_opt, X_test, y_test)
    print("v_net_opt error test set: {}".format(test['particle']['error']))


# seed(1)

# choose dataset
dataset = Datasets.load_cancer()

classes = set([example[-1] for example in dataset])

print("training examples: {}".format(len(dataset)))
print("classes: {}".format(classes))

x = [example[0:len(example)-1] for example in dataset]
y = [example[-1] for example in dataset]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

v_net_opt, output_by_iteration = MLP_PSO_W.run(X_train, X_val, y_train, y_val,
                                             n_particles=30, n_hidden=7,
                                             n_output=len(classes), max_iter=5000,
                                             v_lim=[-1, 1], p_lim=[-1, 1])

print_mlp_pso_results(v_net_opt, output_by_iteration, "output_mlp_pso_w_cancer.csv", MLP_PSO_W)
mlp_pso_in_test_set(v_net_opt, X_test, y_test, MLP_PSO_W)


'''
# Para executar o backpropagation remover o bloco de comentario

# evaluate algorithm
n_folds = 5
l_rate = 0.2
n_epoch = 500
n_hidden = 3

scores = Evaluate.evaluate_algorithm(dataset, Backpropagation.backpropagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
'''