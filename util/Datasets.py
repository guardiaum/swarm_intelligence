from csv import reader
from sklearn.datasets import load_diabetes, load_wine, load_digits
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd


# carrega arquivo csv
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def load_diabetes_as_list():
    dataset = load_diabetes()

    data = dataset.get('data')
    label = dataset.get('target')

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(data)
    data = X.tolist()

    dataset = list()

    for i in range(len(data)):
        aux = list()
        for d in data[i]:
            aux.append(d)
        aux.append(label[i])
        dataset.append(aux)

    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)

    return dataset


def load_digits_as_list():
    dataset = load_digits()

    data = dataset.get('data')
    label = dataset.get('target')

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(data)
    data = X.tolist()

    dataset = list()

    for i in range(len(data)):
        aux = list()
        for d in data[i]:
            aux.append(d)
        aux.append(label[i])
        dataset.append(aux)

    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)

    return dataset


def load_wine_as_list():
    dataset = load_wine()

    data = dataset.get('data')
    label = dataset.get('target')

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(data)
    data = X.tolist()

    dataset = list()

    for i in range(len(data)):
        aux = list()
        for d in data[i]:
            aux.append(d)
        aux.append(label[i])
        dataset.append(aux)

    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)

    return dataset


def load_heart():
    dataset1 = pd.DataFrame(load_csv('datasets/Heart-Data/processed.cleveland.data'))
    dataset2 = pd.DataFrame(load_csv('datasets/Heart-Data/processed.hungarian.data'))
    dataset3 = pd.DataFrame(load_csv('datasets/Heart-Data/processed.switzerland.data'))
    dataset4 = pd.DataFrame(load_csv('datasets/Heart-Data/processed.va.data'))

    dataset = pd.concat([dataset1, dataset2, dataset3, dataset4])

    dataset = tranform_data(dataset)

    return dataset


def tranform_data(dataset):
    dataset = dataset.replace('?', np.nan)
    dataset = dataset.dropna()
    #imputer = Imputer("NaN", strategy="median", axis=0)

    X_data = dataset[dataset.columns[0:len(dataset.columns) - 1]]
    y_data = dataset[dataset.columns[-1]]

    #X_data = imputer.fit_transform(X_data)

    min_max_scaler = preprocessing.MinMaxScaler()

    X = min_max_scaler.fit_transform(X_data)

    data = X.tolist()

    unique_labels = set(y_data)
    lookup = dict()
    labels = list()
    for i, value in enumerate(unique_labels):
        lookup[value] = i
    for value in y_data:
        labels.append(lookup[value])

    for i, d in enumerate(data):
        d.append(labels[i])

    return data


# 2 classes
def load_cancer():
    dataset = load_csv('datasets/breast-cancer.data')

    str_column_to_int(dataset, 0)
    str_column_to_int(dataset, 1)
    str_column_to_int(dataset, 2)
    str_column_to_int(dataset, 3)
    str_column_to_int(dataset, 4)
    str_column_to_int(dataset, 5)
    str_column_to_float(dataset, 6)
    str_column_to_int(dataset, 7)
    str_column_to_int(dataset, 8)
    str_column_to_int(dataset, 9)

    dataset = [[x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[0]] for x in dataset]

    # normalize input variables
    minmax = dataset_minmax(dataset)

    return normalize_dataset(dataset, minmax)


# 3 classes
def load_seeds():
    dataset = load_csv('datasets/seeds_dataset.txt')
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    return normalize_dataset(dataset, minmax)


# 3 classes
def load_iris():
    dataset = load_csv('datasets/iris.data')
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    return normalize_dataset(dataset, minmax)


# converte colunas com strings para float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# converte colunas com strings para inteiros
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Encontra o máximo e mínimo de cada coluna
# Utilizado para normalizar os valores de entrada
def dataset_minmax(dataset):
    return [[min(column), max(column)] for column in zip(*dataset)]


# Normaliza as columnas do dataset para ficarem no intervalo 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset