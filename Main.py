from random import seed
import Datasets
import Evaluate
from optimization_algorithms import Backpropagation
from optimization_algorithms.GlobalSwarm import *
import MLP
import MLP_PSO

seed(1)

# choose dataset
dataset = Datasets.load_iris()

classes = set([example[-1] for example in dataset])

print("training examples: {}".format(len(dataset)))
print("classes: {}".format(classes))

# evaluate algorithm
n_folds = 5
l_rate = 0.2
n_epoch = 500
n_hidden = 3

MLP_PSO.run(dataset, n_particles=100, n_hidden=5, n_output=len(classes), max_iter=500, v_lim=[-1, 1], p_lim=[-1, 1])

'''
# Para executar o backpropagation remover o bloco de comentario
scores = Evaluate.evaluate_algorithm(dataset, Backpropagation.backpropagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
'''