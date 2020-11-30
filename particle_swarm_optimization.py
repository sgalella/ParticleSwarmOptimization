import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)
RES = 200

MIN_X = -5
MAX_X = 5
MIN_Y = -3
MAX_Y = 3
MAX_VEL = 0.01
MIN_VEL = -0.01

MAX_ITERATIONS = 100


def get_landscape_cost(x, y):
    """
    Generates a random landscape.

    Args:
        x (np.array): Meshgrid for x coordinate.
        y (np.array): Meshgrid for y coordinate.

    Returns:
        np.array: Array with costs for each coordinate.
    """
    return x ** 2 + y ** 2  # Sphere
    # return 1 + (x ** 2 / 4000) + (y ** 2 / 4000) - np.cos(x / np.sqrt(2)) - np.cos(y / np.sqrt(2))  # Gricwank
    # return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2  # Himmelblau
    # return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) + np.exp(1) + 20   # Ackley
    # return 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) - 10 * np.cos(2 * np.pi * y)  # Rastrigin


def generate_landscape():
    """
    Generates the landscape of the cost fitness.
    Returns:
        tuple: Coordinates of the landscape with the fitness of each position
    """
    x = np.linspace(MIN_X, MAX_X, RES)
    y = np.linspace(MIN_Y, MAX_Y, RES)
    X, Y = np.meshgrid(x, y)
    Z = get_landscape_cost(X, Y)
    return X, Y, Z


class SwarmOptimizationAlgorithm:
    """ Runs the swarm optimization in a landscape """
    def __init__(self, X, Y, Z, N, omega=0.05, phi_p=0.05, phi_g=0.5):
        """
        Initializes the algorithm.

        Args:
            X (np.array): x-coordinate.
            Y (np.array): y-coordinate.
            Z (np.array): Cost at each location.
            N (int): Number of particles in swarm.
            omega (float, optional): Individual velocity step. Defaults to 0.05.
            phi_p (float, optional): Individual best position step. Defaults to 0.05.
            phi_g (float, optional): Swarm best position step. Defaults to 0.5.
        """
        self.X = X
        self.Y = Y
        self.Z = Z
        self.N = N
        self.Z = Z
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.best_position = None
        self.best_cost = 0
        self.particles = []
        for i in range(N):
            initial_pos_x = np.random.uniform(MIN_X, MAX_X)
            initial_pos_y = np.random.uniform(MIN_Y, MAX_Y)
            initial_cost = self.calculate_cost(initial_pos_x, initial_pos_y, self.Z)
            new_particle = Particle(initial_pos_x, initial_pos_y, initial_cost, self)
            self.update_best_position(new_particle)
            self.particles.append(new_particle)
        self.update_total_cost()

    def __repr__(self):
        """ Print parameter information """
        return "\n".join(f"{particle}" for particle in self.particles)

    def get_coordinates(self, pos, array):
        """
        Gets closest coordinates in landscape.

        Args:
            pos (float): Current position.
            array (np.array): Meshgrid coordinates.

        Returns:
            int: Closest value in landscape.
        """
        return np.unravel_index((np.abs(array - pos)).argmin(), array.shape)

    def calculate_cost(self, pos_x, pos_y, Z):
        """
        Maps current position to the closest point in the landscape.

        Args:
            pos_x (float): x-coordinate position.
            pos_y (float): y-coordinate position.
            Z (np.array): Cost landscape.

        Returns:
            float: Cost of particle at position (pos_x, pos_y)
        """
        _, j = self.get_coordinates(pos_x, self.X)
        i, _ = self.get_coordinates(pos_y, self.Y)
        return Z[i, j]

    def update_total_cost(self):
        """ Calculates the total cost of particles """
        self.total_cost = 0
        for particle in self.particles:
            self.total_cost += particle.cost
    
    def update_best_position(self, particle):
        """
        Updates the best position in the swarm.

        Args:
            particle (Particle): Particle object.
        """
        if self.best_position is None:
            self.best_position = (particle.pos_x, particle.pos_y)
            self.best_cost = particle.cost
        else:
            if particle.cost < self.best_cost:
                self.best_position = (particle.pos_x, particle.pos_y)
                self.best_cost = particle.cost

    def run(self):
        """ Runs the swarm optimization algorithm """
        plt.figure(figsize=(8, 5))
        plt.ion()
        cs = plt.contour(self.X, self.Y, self.Z)
        plt.clabel(cs, inline=1, fontsize=6)
        plt.imshow(self.Z, extent=[MIN_X, MAX_X, MIN_Y, MAX_Y], origin="lower", alpha=0.3)
        plt.colorbar(shrink=0.75)
        self.plot()
        plt.title(f"Total cost: {self.total_cost}")
        for _ in range(MAX_ITERATIONS):
            for particle in self.particles:
                r_px, r_py = np.random.random((2, ))
                r_gx, r_gy = np.random.random((2, ))
                particle.vel_x = self.omega * particle.vel_x + self.phi_p * r_px * (particle.best_pos_x - particle.pos_x) \
                                                             + self.phi_g * r_gx * (self.best_position[0] - particle.pos_x)
                particle.vel_y = self.omega * particle.vel_y + self.phi_p * r_py * (particle.best_pos_y - particle.pos_y) \
                                                             + self.phi_g * r_gy * (self.best_position[1] - particle.pos_y)
                particle.pos_x += particle.vel_x
                particle.pos_y += particle.vel_y
                particle.cost = self.calculate_cost(particle.pos_x, particle.pos_y, self.Z)
                cost_best_pos = self.calculate_cost(particle.best_pos_x, particle.best_pos_y, self.Z)
                if particle.cost < cost_best_pos:
                    particle.best_pos_x = particle.pos_x
                    particle.best_pos_y = particle.pos_y
                    if particle.cost < self.best_cost:
                        self.best_position = (particle.best_pos_x, particle.best_pos_y)
            self.update_total_cost()
            plt.cla()
            cs = plt.contour(self.X, self.Y, self.Z, 10)
            plt.clabel(cs, inline=1, fontsize=8)
            plt.imshow(self.Z, extent=[MIN_X, MAX_X, MIN_Y, MAX_Y], origin="lower", alpha=0.3)
            self.plot()
            plt.title(f"Total cost: {self.total_cost:.2f}")
            plt.draw()
            plt.pause(0.1)
        plt.ioff()
        plt.show()

    def plot(self):
        """ Plots the swarm in the landscape """
        for i in range(self.N):
            self.particles[i].plot()


class Particle:
    """ Individual particles """
    def __init__(self, x, y, initial_cost, swarm):
        """
        Initializes particle.

        Args:
            x (float): x-coordinate position.
            y (float): y-coordinate position.
            initial_cost (float): Initial cost of particle at current position.
            swarm (Swarm): Swarm the particle belongs to.
        """
        self.pos_x = x
        self.pos_y = y
        self.cost = initial_cost
        self.best_pos_x = self.pos_x
        self.best_pos_y = self.pos_y
        if swarm.best_cost:
            if self.cost < swarm.best_cost:
                swarm.best_position = (self.best_pos_x, self.best_pos_y)
                swarm.best_cost = self.cost
        self.vel_x = np.random.uniform(MIN_VEL, MAX_VEL) 
        self.vel_y = np.random.uniform(MIN_VEL, MAX_VEL)

    def __repr__(self):
        """ Print parameter information """
        return (f"Particle(({self.pos_x:.2f}, {self.pos_y:.2f}), ({self.vel_x:.2f}, {self.vel_y:.2f}),"
                f"({self.best_pos_x:.2f}, {self.best_pos_y:.2f}), {self.cost:.2f})")

    def plot(self):
        """ Plots the particle in landscape"""
        plt.plot(self.pos_x, self.pos_y, 'r*')
        plt.arrow(self.pos_x, self.pos_y, self.vel_x, self.vel_y, width=0.02, color='r')


def main():
    X, Y, Z = generate_landscape()
    swarm = SwarmOptimizationAlgorithm(X, Y, Z, 30)
    swarm.run()


if __name__ == "__main__":
    main()
