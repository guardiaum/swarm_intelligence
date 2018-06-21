from beans.ParticleFIPS import ParticleFIPS
from random import uniform


class FIPSSwarm(object):

    def __init__(self, function, bounds, swarm_size, nsize, max_iter, w_type, inertia_w=0.7298):
        self.function = function
        self.bounds = bounds
        self.dimensions = len(bounds)
        self.swarm_size = swarm_size
        self.nsize = nsize
        self.max_iter = max_iter
        self.w_type = w_type
        self.inertia_w = inertia_w

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

            swarm.append(ParticleFIPS(p0, v0))
            # print("P: %s > %s - V: %s" % (i, swarm[i].position, swarm[i].velocity))

        return swarm

    def main(self):

        gbest = []
        error_best = -1
        error_bests = []

        swarm = self.initialize_swarm(self.bounds, self.dimensions, self.swarm_size)

        for i in range(0, len(swarm)):
            swarm[i].evaluate(self.function)

        iter = 0
        while iter < self.max_iter:

            for i in range(0, len(swarm)):
                swarm[i].get_neighbors(i, self.nsize, swarm)

                swarm[i].update_velocity(self.inertia_w, self.dimensions, self.w_type)

                swarm[i].update_position(self.bounds, self.dimensions)

                swarm[i].evaluate(self.function)

                if swarm[i].error < error_best or error_best == -1:
                    gbest = swarm[i].position
                    error_best = swarm[i].error_best

            error_bests.append(error_best)
            iter += 1

        # print("FIPS Model - >>> gBest: %s - error: %s" % (gbest, error_best))
        return error_best, error_bests
