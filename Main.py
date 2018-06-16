from random import seed
import Datasets
import Evaluate
from optimization_algorithms import Backpropagation
from optimization_algorithms.GlobalSwarm import *
import MLP
import MLP_PSO

# Test Backprop on Seeds dataset
seed(1)

dataset = Datasets.load_iris()

print("training examples: {}".format(len(dataset)))

# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 100
n_hidden = 5

MLP_PSO.run(dataset, n_particles=30, n_hidden=5, n_output=3, max_iter=1000)

'''
# Para executar o backpropagation remover o bloco de comentario
scores = Evaluate.evaluate_algorithm(dataset, Backpropagation.backpropagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
'''