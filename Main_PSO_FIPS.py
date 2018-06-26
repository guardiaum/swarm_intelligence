from util import Results, Datasets
from sklearn.model_selection import train_test_split
from optimization_algorithms import MLP_PSO_FIPS


dataset = Datasets.load_digits_as_list()

classes = set([example[-1] for example in dataset])

print("training examples: {}".format(len(dataset)))
print("classes: {}".format(classes))

x = [example[0:len(example)-1] for example in dataset]
y = [example[-1] for example in dataset]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

v_net_opt, output_by_iteration = MLP_PSO_FIPS.run(X_train, X_val, y_train, y_val,
        n_particles=30, n_hidden=7, n_output=len(classes),
        max_iter=2000, check_gloss=500, neighborhood_size=2, weight_method='distance',
        inertia_weight=0.7298, p_lim=[-1, 1])

Results.print_results(v_net_opt, output_by_iteration, "results/output_mlp_pso_fips_cancer.csv", MLP_PSO_FIPS)
Results.run_in_test_set(v_net_opt, X_test, y_test, MLP_PSO_FIPS)
