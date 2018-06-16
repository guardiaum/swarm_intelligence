from random import random

class Particle(object):

    def __init__(self, p0, v0):
        self.position = p0
        self.velocity = v0
        self.pbest = []
        self.error_best = -1
        self.error = -1

    def evaluate(self, function):

        self.error = function(self.position)

        if self.error < self.error_best or self.error_best == -1:
            self.pbest = self.position
            self.error_best = self.error

        # print("FIT ", self.error)
        # print("BEST FIT ", self.error_best)

    def update_velocity(self, nbest, dimensions, inertia_w, cognitive_c1, social_c2):
        for i in range(0, dimensions):
            r1 = random()
            r2 = random()

            cognitive_vel = cognitive_c1 * r1 * (self.pbest[i] - self.position[i])
            social_vel = social_c2 * r2 * (nbest[i] - self.position[i])
            self.velocity[i] = inertia_w * self.velocity[i] + cognitive_vel + social_vel

    def update_position(self, bounds, dimensions):
        for i in range(0, dimensions):
            self.position[i] = self.position[i] + self.velocity[i]

            if self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]

        # print("POSITION ", self.position)