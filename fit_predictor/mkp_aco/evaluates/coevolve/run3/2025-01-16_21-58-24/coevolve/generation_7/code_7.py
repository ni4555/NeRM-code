import numpy as np
import numpy as np
from scipy.optimize import differential_evolution

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Define the fitness function
    def fitness(individual):
        total_prize = np.dot(individual, prize)
        if not np.all(np.sum(individual * weight, axis=1) <= 1):
            return -np.inf
        return total_prize

    # Define the bounds for the differential evolution
    bounds = [(0, 1) for _ in range(len(prize))]

    # Run the differential evolution algorithm
    result = differential_evolution(fitness, bounds, strategy='best1bin', maxiter=100, popsize=50)

    # Convert the result to a probability distribution
    max_fitness = np.max(result.fun)
    probabilities = np.exp((result.fun - max_fitness) / (max_fitness - np.min(result.fun)))
    probabilities /= np.sum(probabilities)

    # Convert probabilities to a shape (n,) array
    heuristics = probabilities * 100  # Scale the probabilities to a range of [0, 100]

    return heuristics