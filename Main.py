import csv
from util import Datasets, Results
from optimization_algorithms import Backpropagation
from sklearn.model_selection import train_test_split
from optimization_algorithms import MLP_CS_W
from beans import function as fn
from optimization_algorithms import MLP_PSO_Classic
from optimization_algorithms import MLP_PSO_FIPS
from optimization_algorithms import MLP_PSO_RING
from optimization_algorithms import MLP_PSO_W
import sys
import timeit


#Parse common parameters
dataset_name = sys.argv[sys.argv.index("--dataset") + 1]
algorithm = sys.argv[sys.argv.index("--alg") + 1]
n_hidden = int(sys.argv[sys.argv.index("--hidden") + 1])
max_iter = int(sys.argv[sys.argv.index("--maxiter") + 1])
number_experiments = int(sys.argv[sys.argv.index("--trial") + 1])
fips_method = ""
dataset = ""

if dataset_name in ["wine", "digits"]:
	dataset = eval("Datasets.load_%s_as_list()"%dataset_name)

else:
	dataset = eval("Datasets.load_%s()"%dataset_name)

classes = set([example[-1] for example in dataset])

print("training examples: {}".format(len(dataset)))
print("classes: {}".format(classes))

x = [example[0:len(example)-1] for example in dataset]
y = [example[-1] for example in dataset]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

arguments = ""
results = []

for i in range(number_experiments):
	print ("\nITERATION %d: "%(i+1))
	if algorithm == "cs":
		p = int(sys.argv[sys.argv.index("--p") + 1])
		check_gloss = int(sys.argv[sys.argv.index("--gloss") + 1])
		t1 = timeit.default_timer()
		v_net_opt, output_by_iteration = MLP_CS_W.run(X_train, X_val, y_train, y_val, n_eggs=p, max_iter = max_iter, check_gloss=check_gloss, n_hidden=n_hidden, n_output=len(classes))
		t2 = timeit.default_timer()
		Results.print_results_cs(v_net_opt, output_by_iteration, "results/output_mlp_cs_%s_%d.csv"%(dataset_name, i+1), fn)
		results.append ([Results.run_in_test_set_cs(v_net_opt, X_test, y_test, fn), t2-t1])
		print ("runtime: %f s"%(t2-t1))

	elif algorithm == "bp":
		#Parse parameters
		l_rate = float(sys.argv[sys.argv.index("--rate") + 1])
		
		for x in range(len(X_train)):
			X_train[x].append(y_train[x])

		for x in range(len(X_test)):
			X_test[x].append(y_test[x])

		for x in range(len(X_val)):
			X_val[x].append(y_val[x])
		
		t1 = timeit.default_timer()	
		network, output_by_iteration = Backpropagation.backpropagation(X_train, l_rate, max_iter, n_hidden, n_outputs=len(classes))
		t2 = timeit.default_timer()
		
		with open('results/output_backpropagation_%s_%d.csv'%(dataset_name, i+1), 'w') as f:
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
		
		results.append ([accuracy, t2-t1])
		
	else:
		p = int(sys.argv[sys.argv.index("--p") + 1])
		check_gloss = int(sys.argv[sys.argv.index("--gloss") + 1])
		if algorithm == "pso":
			#Parse parameters
			inertia_weight = float(sys.argv[sys.argv.index("--inertia") + 1])
			c1 = float(sys.argv[sys.argv.index("--c1") + 1])
			c2 = float(sys.argv[sys.argv.index("--c2") + 1])
			method = MLP_PSO_Classic

			t1 = timeit.default_timer()	
			v_net_opt, output_by_iteration = MLP_PSO_Classic.run(X_train, X_val, y_train, y_val,
														 n_particles=p, n_hidden=n_hidden,
														 n_output=len(classes), 
														 max_iter=max_iter, check_gloss=check_gloss,
														 inertia_weight=inertia_weight,
														 c1=c1, c2=c2,
														 v_lim=[-1, 1], p_lim=[-1, 1])
			t2 = timeit.default_timer()	

		elif algorithm == "fips":
			inertia_weight = float(sys.argv[sys.argv.index("--inertia") + 1])
			neighborhood_size = int(sys.argv[sys.argv.index("--k") + 1])
			weight_method = sys.argv[sys.argv.index("--wmethod") + 1]
			method = MLP_PSO_FIPS

			t1 = timeit.default_timer()	
			v_net_opt, output_by_iteration = MLP_PSO_FIPS.run(X_train, X_val, y_train, y_val,
			        n_particles=p, n_hidden=n_hidden, n_output=len(classes), 
					max_iter=max_iter, check_gloss=check_gloss, neighborhood_size=neighborhood_size, 
					weight_method=weight_method, inertia_weight=inertia_weight, p_lim=[-1, 1])

			t2 = timeit.default_timer()	

			fips_method = "_%s"%(weight_method)


		elif algorithm == "ring":
			inertia_weight = float(sys.argv[sys.argv.index("--inertia") + 1])
			neighborhood_size = int(sys.argv[sys.argv.index("--k") + 1])
			c1 = float(sys.argv[sys.argv.index("--c1") + 1])
			c2 = float(sys.argv[sys.argv.index("--c2") + 1])
			method = MLP_PSO_RING

			t1 = timeit.default_timer()	
			v_net_opt, output_by_iteration = MLP_PSO_RING.run(X_train, X_val, y_train, y_val,
                    n_particles=p, n_hidden=n_hidden, n_output=len(classes), 
					max_iter=max_iter, check_gloss=check_gloss, neighborhood_size=neighborhood_size, 
					inertia_weight=inertia_weight, c1=c1, c2=c2, v_lim=[-1, 1], p_lim=[-1, 1])

			t2 = timeit.default_timer()

		elif algorithm == "psow":
			c1 = float(sys.argv[sys.argv.index("--c1") + 1])
			c2 = float(sys.argv[sys.argv.index("--c2") + 1])
			method = MLP_PSO_RING

			t1 = timeit.default_timer()	

			v_net_opt, output_by_iteration = MLP_PSO_W.run(X_train, X_val, y_train, y_val,
                                            n_particles=p, n_hidden=n_hidden,
											n_output=len(classes), 
											max_iter=max_iter, check_gloss=check_gloss,
                                            v_lim=[-1, 1], p_lim=[-1, 1])
			t2 = timeit.default_timer()	
		
		Results.print_results(v_net_opt, output_by_iteration, "results/output_mlp_%s%s_%s_%d.csv"%(algorithm, fips_method, dataset_name, i+1), method)
		results.append ([Results.run_in_test_set(v_net_opt, X_test, y_test, MLP_PSO_Classic), t2-t1])
		print ("runtime: %f s"%(t2-t1))

best = 0.0
best_time = float("inf")

with open('results/results_%s%s_%s.csv'%(algorithm, fips_method, dataset_name), 'w') as f:
	writer = csv.writer(f)
	for accurancy, runtime in results:
		writer.writerow([accurancy, runtime])
		if (accurancy > best) or (accurancy == best and runtime < best_time):
			best = accurancy
			best_time = runtime

print ("\nBest of 30: Iteration %d"% (results.index([best, best_time])+1))