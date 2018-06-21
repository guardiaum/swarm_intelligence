from util import Datasets
from util import Evaluate
from optimization_algorithms import Backpropagation

n_folds = 5
l_rate = 0.2
n_epoch = 500
n_hidden = 3

# choose dataset
dataset = Datasets.load_cancer()

scores = Evaluate.evaluate_algorithm(dataset, Backpropagation.backpropagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))