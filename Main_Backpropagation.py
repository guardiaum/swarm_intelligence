import csv
from util import Datasets
from optimization_algorithms import Backpropagation
from sklearn.model_selection import train_test_split


n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 3

# choose dataset
dataset = Datasets.load_digits_as_list()

classes = set([example[-1] for example in dataset])

print("training examples: {}".format(len(dataset)))
print("classes: {}".format(classes))

x = [example[0:len(example)-1] for example in dataset]
y = [example[-1] for example in dataset]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

for x in range(len(X_train)):
    X_train[x].append(y_train[x])

for x in range(len(X_test)):
    X_test[x].append(y_test[x])

for x in range(len(X_val)):
    X_val[x].append(y_val[x])


network, output_by_iteration = Backpropagation.backpropagation(X_train, l_rate, n_epoch, n_hidden, n_outputs=len(classes))

with open('results/output_backpropagation_cancer.csv', 'w') as f:
    writer = csv.writer(f)
    for iteration, error, hidden, conn in output_by_iteration:
        writer.writerow([iteration, error, hidden, conn])

error = 0
for i, row in enumerate(X_test):
    prediction = Backpropagation.predict(network, row)
    if prediction != y_test[i]:
        error += 1

error = error / len(X_test)
accuracy = (1 - error) * 100
print("v_net_opt error on test set: {}".format(error))
print("v_net_opt accuracy on test set: {}%".format(accuracy))