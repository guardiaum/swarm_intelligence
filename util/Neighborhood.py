import math


class Neighborhood(object):

    @staticmethod
    def euclidian_dist(particle_position, neighbor_position):
        distance = 0

        for i in range(0, len(particle_position)):
            distance += (particle_position[i] - neighbor_position[i]) ** 2

        return math.sqrt(distance)

    @staticmethod
    def get_static(target_index, n_size, swarm_size):
        neighbors = []
        range_by_side = n_size // 2

        for i in range(1, range_by_side + 1):
            previous = target_index - i
            next = target_index + i

            if next >= swarm_size:
                next = -1 + i

            neighbors.append(previous)
            neighbors.append(next)

        return neighbors

    @staticmethod
    def get_dynamic(particle_position, n_size, swarm):

        def use_distance(temp):
            return temp[1]

        distances = []

        for i in range(len(swarm)):
            neighbor = swarm[i]
            distance = Neighborhood.euclidian_dist(particle_position, neighbor.position)
            distances.append([i, distance])

        distances = sorted(distances, key=use_distance)

        neighbors = []
        for j in range(1, n_size + 1):
            neighbors.append(swarm[distances[j][0]])

        return neighbors
