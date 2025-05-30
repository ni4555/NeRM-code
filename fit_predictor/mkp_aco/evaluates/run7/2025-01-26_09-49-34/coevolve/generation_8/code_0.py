import numpy as np
import numpy as np
import random

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape
    num_knapsacks = m  # Since each dimension has a weight constraint of 1
    probability_threshold = 0.5

    def fitness(solution):
        total_weight = np.sum([np.sum(weight[sol, :]) for sol in solution])
        return total_weight, np.sum(prize[sol] for sol in solution if sol not in solution)

    initial_solution = list(range(n))
    max_solution = initial_solution.copy()
    max_fitness = fitness(initial_solution)

    for _ in range(1000):  # Number of iterations
        random.shuffle(initial_solution)
        candidate_solution = [initial_solution[i] for i in range(n) if random.random() > probability_threshold]

        current_fitness = fitness(candidate_solution)
        if current_fitness > max_fitness:
            max_solution = candidate_solution
            max_fitness = current_fitness

    heuristic = np.zeros(n)
    heuristic[max_solution] = 1
    return heuristic
