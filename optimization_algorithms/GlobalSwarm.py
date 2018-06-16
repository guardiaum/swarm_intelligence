from beans.Particle import Particle
from random import random
from random import uniform


class GlobalSwarm(object):

    def __init__(self, function, bounds, swarm_size, max_iter,
                 inertia_w=0.5, cognitive_c1=0.5, social_c2=0.5):
        self.function = function
        self.bounds = bounds
        self.dimensions = len(bounds)
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.inertia_w = inertia_w
        self.cognitive_c1 = cognitive_c1
        self.social_c2 = social_c2

    @staticmethod
    def initialize_swarm(bounds, dimensions, swarm_size):
        swarm = []

        # print("INITIAL POSITIONING")

        lower_bound = bounds[0][0]
        upper_bound = bounds[0][1]

        for i in range(0, swarm_size):

            p0 = []
            v0 = []
            for dim in range(0, dimensions):
                p0.append(uniform(lower_bound, upper_bound))
                v0.append(uniform(lower_bound, upper_bound))

            swarm.append(Particle(p0, v0))
            # print("p: %s -> %s" % (i, swarm[i].position))

        return swarm

    def main(self):

        error_gbest = -1
        error_gbests = []
        gbest = []

        swarm = self.initialize_swarm(self.bounds, self.dimensions, self.swarm_size)

        i = 0
        while i < self.max_iter:

            # print("EVALUATE ERROR")
            for j in range(0, self.swarm_size):
                swarm[j].evaluate(self.function)
                # print("p: %s -> %s -> error: %s" % (j, swarm[j].position, swarm[j].error))

                if swarm[j].error < error_gbest or error_gbest == -1:
                    gbest = list(swarm[j].position)
                    error_gbest = float(swarm[j].error)

            # print("UPDATE VELOCITY AND POSITION")
            for j in range(0, self.swarm_size):
                swarm[j].update_velocity(gbest, self.dimensions, self.inertia_w, self.cognitive_c1, self.social_c2)
                swarm[j].update_position(self.bounds, self.dimensions)
                # print("p: %s, pos ->%s\n-> vel:%s" % (j, swarm[j].position, swarm[j].velocity))

            error_gbests.append(error_gbest)
            i += 1

            # print("ITERATION: %s" % i)
            # print("gBest: %s - error: %s" % (gbest, error_gbest))

        # print("gBest Model - >>> gBest: %s - error: %s" % (gbest, error_gbest))
        return error_gbest, error_gbests
