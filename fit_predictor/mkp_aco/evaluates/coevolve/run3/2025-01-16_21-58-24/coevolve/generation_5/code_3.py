import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the fitness for each item based on the weighted combination of values
    # and adherence to the multi-dimensional constraints
    n, m = prize.shape[0], weight.shape[1]
    fitness = np.zeros(n)
    for i in range(n):
        # Calculate the weighted value of the item
        weighted_value = np.sum(prize[i] * weight[i])
        # Calculate the adherence to constraints (assuming constraint is 1 for each dimension)
        adherence = np.sum(weight[i] == 1)
        # Fitness is a combination of weighted value and adherence
        fitness[i] = weighted_value + adherence
    
    # Normalize the fitness scores to make them more comparable
    max_fitness = np.max(fitness)
    min_fitness = np.min(fitness)
    if max_fitness - min_fitness > 0:
        fitness = (fitness - min_fitness) / (max_fitness - min_fitness)
    else:
        fitness = np.ones(n)
    
    return fitness