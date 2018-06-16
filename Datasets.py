from csv import reader

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

def load_seeds():
    global dataset
    dataset = load_csv('datasets/seeds_dataset.txt')
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    return normalize_dataset(dataset, minmax)

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