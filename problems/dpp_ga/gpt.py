import numpy as np
import numpy as np

def crossover_v2(parents: np.ndarray, n_pop: int) -> np.ndarray:
    crossover_points = np.random.randint(1, parents.shape[1], size=2)
    crossover_points = np.sort(crossover_points)
    
    offspring = np.empty((n_pop, parents.shape[1]))
    for i in range(n_pop):
        parent1 = parents[np.random.randint(parents.shape[0])]
        parent2 = parents[np.random.randint(parents.shape[0])]
        offspring[i] = np.concatenate([parent1[:crossover_points[0]], parent2[crossover_points[0]:crossover_points[1]], parent1[crossover_points[1]:]])
    
    return offspring
