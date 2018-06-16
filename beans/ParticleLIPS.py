from beans.Particle import Particle
from util.Neighborhood import Neighborhood
from random import uniform


class ParticleLIPS(Particle):

    def __init__(self, p0, v0):
        Particle.__init__(self, p0, v0)
        self.neighbors = []
        self.n_size = len(self.neighbors)

    def update_velocity(self, inertia_w, dimensions):
        phi = self.calculate_phi()

        p = self.calculate_p(phi, dimensions)

        phi_ = 0
        for i in range(0, len(phi)):
            phi_ += phi[i]

        for d in range(0, dimensions):
            self.velocity[d] = inertia_w * (self.velocity[d] + phi_ * (p[d] - self.position[d]))

    def calculate_p(self, phi, dimensions):
        p = []
        numerator = 0
        phi_ = 0

        for d in range(0, dimensions):
            for k in range(0, len(self.neighbors)):
                numerator += (phi[k] * self.neighbors[k].position[d]) / self.n_size
                phi_ += phi[k]

            p.append(numerator / phi_)

        return p

    def calculate_phi(self):

        phi_limit = 4.1 / self.n_size
        phi = []

        x = uniform(0, phi_limit)

        for i in range(0, self.n_size):
            phi.append(x)

        return phi

    def get_neighbors(self, n_size, swarm):
        self.neighbors = Neighborhood.get_dynamic(self.position, n_size, swarm)
        self.n_size = len(self.neighbors)
