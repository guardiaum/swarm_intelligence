from random import random
from beans.config import Config
from beans import individual as id
import numpy as np
import copy
import beans.function as fn


def run(X_train, X_val, y_train, y_val, n_eggs, max_iter, check_gloss, n_hidden, n_output):
    cf = Config()

    # input neurons
    n_input = len(X_train[0])

    # initialize population
    #n_eggs = cf.get_population_size()
    population = [[]] * n_eggs
    for egg in range(n_eggs):
        population[egg] = id.Individual(cf, n_input, n_hidden, n_output, X_train, y_train)

    population = sorted(population, key=lambda ID: ID.get_fitness())

    Bestnet = population[0].get_net()
    BestFitness = population[0].get_fitness()

    iteration = 0
    stagnation_count = 0
    egg_loss = 0
    best_egg = {"error": float('inf')}
    v_net_opt = None
    hist = []
    output_by_iteration = []

    while iteration < max_iter and egg_loss < 5 and stagnation_count < 3:

    
    #for iteration in range(max_iter):
        ##print("iteration: {}".format(iteration))
        # computes eggs local best (pBest) with lower training error

        for i in range(len(population)):
            population[i].get_cuckoo()
            population[i].set_fitness()

            """random choice (say j)"""
            j = np.random.randint(low=0, high=cf.get_population_size())
            while j == i: #random id[say j] =/= i
                j = np.random.randint(0, cf.get_population_size())

                # for minimize problem
            if(population[i].get_fitness() < population[j].get_fitness()):
                population[j].set_net(population[i].get_net())

        """Sort (to Keep Best)"""
        population = sorted(population, key=lambda ID: ID.get_fitness())

        """Abandon Solutions (exclude the best)"""
        for a in range(1,len(population)):
            r = np.random.rand()
            if(r < cf.get_Pa()):
                population[a].abandon()
                population[a].set_fitness()

            """Sort to Find the Best"""
        population = sorted(population, key=lambda ID: ID.get_fitness())

        if population[0].get_fitness() < BestFitness:
            BestFitness = population[0].get_fitness()
            Bestnet = copy.deepcopy(population[0].get_net())
            #print("best_error: {}".format(BestFitness))

        #sys.stdout.write("\n\r Trial:%3d , Iteration:%7d, BestFitness:%.4f" % (trial , iteration, BestFitness))

           # results_list.append(str(BestFitness))
        

        #print("best_error: {}".format(BestFitness))

        print("iteration: {}".format(iteration), "best_error: {}".format(BestFitness))

        output_by_iteration.append([iteration, fn.get_iteration_data(Bestnet)])

        hist.append(Bestnet)


        # get error in validation ser for v_net_current and v_net_opt
        if ((iteration > check_gloss) and ((iteration + 1) % 100 == 0)) or iteration == max_iter - 1:

            v_net_current = fn.forward_propagate(Bestnet, X_val, y_val)
            min_net_from_hist = copy.deepcopy(min(hist, key=lambda x: x['error']))
            v_net_opt = fn.forward_propagate(min_net_from_hist, X_val, y_val)

            if v_net_current["error"] < v_net_opt["error"]:
                v_net_opt = copy.deepcopy(v_net_current)
                stagnation_count = 0
            else:
                egg_loss = generalization_loss(v_net_opt, v_net_current)
                stagnation_count = stagnation_count + 1
        iteration += 1

    return v_net_opt, output_by_iteration


# generaliziation loss used to stop execution when reach criteria
def generalization_loss(v_net_opt, v_net_current):
    if v_net_opt['error'] == 0:
        return 0
    else: return 100 * (v_net_current['error'] / v_net_opt['error']) - 1