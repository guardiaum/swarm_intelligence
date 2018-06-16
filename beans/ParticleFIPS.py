from beans.Particle import Particle
from util.Neighborhood import Neighborhood
from random import uniform


class ParticleFIPS(Particle):

    def __init__(self, p0, v0):
        Particle.__init__(self, p0, v0)
        self.neighbors = []
        self.n_size = len(self.neighbors)

    def get_neighbors(self, target, n_size, swarm):
        indexes = Neighborhood.get_static(target, n_size, len(swarm))

        # print("NEIGHBORS", indexes)
        self.neighbors = []
        for i in range(len(indexes)):
            self.neighbors.append(swarm[indexes[i]])

        self.n_size = len(self.neighbors)

    def update_velocity(self, inertia_w, dimensions, w_type):
        phi = self.calculate_phi()
        p = self.calculate_p(phi, dimensions, w_type)

        phi_ = 0
        for i in range(0, len(phi)):
            phi_ += phi[i]

        for d in range(0, dimensions):
            self.velocity[d] = inertia_w * (self.velocity[d] + phi_ * (p[d] - self.position[d]))

    def calculate_p(self, phi, dimensions, w_type):
        p = []
        numerator = 0
        divisor = 0

        for d in range(dimensions):
            for k in range(0, len(self.neighbors)):
                w_value = self.get_w(k, w_type)
                numerator += w_value * phi[k] * self.neighbors[k].pbest[d]
                divisor += w_value * phi[k]

            p.append(numerator / divisor)

        return p

    def get_w(self, k, w_type):
        if w_type == 'static':  # FIPS
            return 0.5
        elif w_type == 'fitness':  # wFIPS
            return self.neighbors[k].error_best
        elif w_type == 'distance':  # wdFIPS
            distance = Neighborhood.euclidian_dist(self.position, self.neighbors[k].pbest)
            if distance < 0.001:
                return 0.001
            else:
                return distance

    def calculate_phi(self):

        phi_limit = 4.1 / self.n_size
        phi = []

        x = uniform(0, phi_limit)

        for i in range(0, self.n_size):
            phi.append(x)

        return phi
