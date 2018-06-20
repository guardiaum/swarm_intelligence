import numpy as np
import math
from random import uniform
from beans.config import Config
import beans.function as fn
import copy


class Individual:
    def __init__(self, cf, n_input, n_hidden, n_output, training_examples, expected_outputs):
        self.cf = cf
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.training_examples = training_examples
        self.expected_outputs = expected_outputs

        H = [0 for i in range(n_hidden)]

        # initialize biases vectors
        B_input = [uniform(-1, 1)  for i in range(n_hidden)]
        B_hidden = [uniform(-1, 1)  for i in range(n_output)]

        # initialize weight matrices
        W_input = [[uniform(-1, 1) for y in range(n_input)] for x in range(n_hidden)]
        W_hidden = [[uniform(-1, 1) for y in range(n_hidden)] for x in range(n_output)]

        """
        H = [None for i in range(n_hidden)]

        # initialize biases vectors
        B_input = [uniform(1, 1) for i in range(n_hidden)]
        B_hidden = [uniform(1, 1) for i in range(n_output)]

        # initialize connections matrices
        C_input = [[uniform(-1, 1) for y in range(n_input)] for x in range(n_hidden)]
        C_hidden = [[uniform(-1, 1) for y in range(n_hidden)] for x in range(n_output)]

        # initialize weight matrices
        W_input = [[uniform(-1, 1) for y in range(n_input)] for x in range(n_hidden)]
        W_hidden = [[uniform(-1, 1) for y in range(n_hidden)] for x in range(n_output)]
"""
        


        self.__net = {'error': None,
                          'hidden': H, "n_output": n_output,
                          "b_input": B_input, "b_hidden": B_hidden,
                          "w_input": W_input, "w_hidden": W_hidden,
                          #"c_input": C_input, "c_hidden": C_hidden,
            }

        #self.__net = np.random.rand(self.cf.get_dimension()) * (self.cf.get_max_domain() - self.cf.get_min_domain())  + self.cf.get_min_domain()
        self.__net = fn.forward_propagate(self.__net, training_examples, expected_outputs)

    def get_net(self):
        return self.__net

    def set_net(self, net):
        self.__net = copy.deepcopy(net)

    def get_fitness(self):
        return self.__net["error"]

    def set_fitness(self):
        self.__net = fn.forward_propagate(self.__net, self.training_examples, self.expected_outputs)

    def abandon(self):
        # abandon some variables
        for i in range(self.n_hidden):
            for j in range(self.n_input):
                p = np.random.rand()
                if p < self.cf.get_Pa():
                    self.__net["w_input"][i][j] = np.random.rand() * (self.cf.get_max_domain() - self.cf.get_min_domain())  + self.cf.get_min_domain()
            p = np.random.rand()
            if p < self.cf.get_Pa():
                self.__net["b_input"][i] = np.random.rand() * (self.cf.get_max_domain() - self.cf.get_min_domain())  + self.cf.get_min_domain()

        for i in range(self.n_output):
            for j in range(self.n_hidden):
                p = np.random.rand()
                if p < self.cf.get_Pa():
                    self.__net["w_hidden"][i][j] = np.random.rand() * (self.cf.get_max_domain() - self.cf.get_min_domain())  + self.cf.get_min_domain()
            p = np.random.rand()
            if p < self.cf.get_Pa():
                self.__net["b_hidden"][i] = np.random.rand() * (self.cf.get_max_domain() - self.cf.get_min_domain())  + self.cf.get_min_domain()

    def get_cuckoo(self):

        step_size_input = self.cf.get_stepsize() * self.levy_flight(self.cf.get_lambda(), self.n_input)
        step_size_hidden = self.cf.get_stepsize() * self.levy_flight(self.cf.get_lambda(), self.n_hidden)
        step_size_output = self.cf.get_stepsize() * self.levy_flight(self.cf.get_lambda(), self.n_output)

        # Update net

        for i in range (self.n_hidden):
            self.__net["w_input"][i] = list(np.array(self.__net["w_input"][i]) + np.array(step_size_input))


        for i in range (len(self.__net["w_hidden"])):
            self.__net["w_hidden"][i] = list(np.array(self.__net["w_hidden"][i]) + np.array(step_size_hidden))

        self.__net["b_input"] = list(np.array(self.__net["b_input"]) + np.array(step_size_hidden))
        self.__net["b_hidden"] = list(np.array(self.__net["b_hidden"]) + np.array(step_size_output))


        # Simple Boundary Rule
        for i in range(self.n_hidden):
            for j in range(self.n_input):
                if self.__net["w_input"][i][j] > self.cf.get_max_domain():
                    self.__net["w_input"][i][j] = self.cf.get_max_domain()
                if self.__net["w_input"][i][j] < self.cf.get_min_domain():
                    self.__net["w_input"][i][j] = self.cf.get_min_domain()

            if self.__net["b_input"][i] > self.cf.get_max_domain():
                self.__net["b_input"][i] = self.cf.get_max_domain()
            if self.__net["b_input"][i] < self.cf.get_min_domain():
                self.__net["b_input"][i] = self.cf.get_min_domain()

        for i in range(self.n_output):
            for j in range(self.n_hidden):
                if self.__net["w_hidden"][i][j] > self.cf.get_max_domain():
                    self.__net["w_hidden"][i][j] = self.cf.get_max_domain()
                if self.__net["w_hidden"][i][j] < self.cf.get_min_domain():
                    self.__net["w_hidden"][i][j] = self.cf.get_min_domain()

            if self.__net["b_hidden"][i] > self.cf.get_max_domain():
                self.__net["b_hidden"][i] = self.cf.get_max_domain()
            if self.__net["b_hidden"][i] < self.cf.get_min_domain():
                self.__net["b_hidden"][i] = self.cf.get_min_domain()

    def print_info(self,i):
        print("id:","{0:3d}".format(i),
              "|| fitness:",str(self.__fitness).rjust(14," "),
              "|| net:",np.round(self.__net,decimals=4))

    def levy_flight(self, Lambda, dimension):
        #generate step from levy distribution
        sigma1 = np.power((math.gamma(1 + Lambda) * np.sin((np.pi * Lambda) / 2)) \
                          / math.gamma((1 + Lambda) / 2) * np.power(2, (Lambda - 1) / 2), 1 / Lambda)
        sigma2 = 1
        u = np.random.normal(0, sigma1, size=dimension)
        v = np.random.normal(0, sigma2, size=dimension)
        step = u / np.power(np.fabs(v), 1 / Lambda)

        return step    # return np.array (ex. [ 1.37861233 -1.49481199  1.38124823])


#if __name__ == '__main__':
#    print(self.levy_flight(self.cf.get_lambda()))

