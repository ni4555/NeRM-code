import numpy as np
import numpy as np
from scipy.optimize import differential_evolution

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Stochastic solution sampling
    np.random.seed(42)
    random_indices = np.random.choice(n, size=10, replace=False)
    random_solutions = weight[random_indices] < 1
    random_solutions = (random_solutions.sum(axis=1) == m).astype(int)
    random_fitness = prize[random_indices] * random_solutions

    # Initialize a population for adaptive evolutionary computation
    population = np.random.rand(20, n)
    population = (population < 0.5).astype(int)
    
    # Adaptive evolutionary computation
    def objective_function(individual):
        if np.any(individual.sum(axis=1) > m) or np.any(individual.sum(axis=1) < m):
            return -np.inf
        else:
            return np.sum(prize[individual.sum(axis=1) == m] * individual)
    
    result = differential_evolution(objective_function, bounds=[(0, 1) for _ in range(n)], strategy='best1bin', maxiter=100, popsize=20, tol=0.01, mutation=(1 / n, 1 / n), recombination=0.8, crosspb=0.9)
    evolutionary_solution = result.x.astype(int)
    evolutionary_fitness = objective_function(evolutionary_solution)

    # Robust local search algorithms
    def local_search(individual, neighborhood_size=3):
        for i in range(n):
            if individual[i] == 0:
                for j in range(1, neighborhood_size + 1):
                    if np.any(weight[i:i+j+1, :].sum(axis=1) > m):
                        break
                    temp = individual.copy()
                    temp[i] = 1
                    if temp.sum(axis=1).max() <= m:
                        yield temp
        for i in range(n):
            if individual[i] == 1:
                for j in range(1, neighborhood_size + 1):
                    if np.any(weight[i-i+j:i+1, :].sum(axis=1) > m):
                        break
                    temp = individual.copy()
                    temp[i] = 0
                    if temp.sum(axis=1).max() <= m:
                        yield temp

    best_local_solution = None
    best_local_fitness = -np.inf
    for new_solution in local_search(evolutionary_solution):
        if np.any(new_solution.sum(axis=1) > m):
            continue
        fitness = objective_function(new_solution)
        if fitness > best_local_fitness:
            best_local_solution = new_solution
            best_local_fitness = fitness

    # Final heuristic values
    combined_fitness = (random_fitness + evolutionary_fitness + best_local_fitness) / 3
    heuristic_values = np.exp(combined_fitness) / np.exp(combined_fitness).sum()

    return heuristic_values