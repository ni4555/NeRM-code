import numpy as np
import numpy as np
import random

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape
    population_size = 100
    elite_size = 10
    mutation_rate = 0.1
    crossover_rate = 0.8
    generations = 50
    local_search_iterations = 10
    
    # Initialize the population with random solutions
    population = np.random.choice([0, 1], size=(population_size, n))
    
    for generation in range(generations):
        # Evaluate the fitness of each individual in the population
        fitness = np.dot(population, prize) - np.sum(weight * population, axis=1)
        
        # Apply stochastic local search to the elite individuals
        for individual in population[:elite_size]:
            for _ in range(local_search_iterations):
                # Randomly select a bit to flip
                bit_to_flip = random.randint(0, n-1)
                # Flip the bit
                individual[bit_to_flip] = 1 - individual[bit_to_flip]
                # Apply a heuristic to improve the individual
                for _ in range(local_search_iterations):
                    if np.sum(weight * individual) <= m and np.sum(individual) < n:
                        for i in range(n):
                            if individual[i] == 0 and np.sum(weight[:i] * individual[:i]) < m:
                                individual[i] = 1
                                break
                # Undo the flip if the new individual is not better
                individual[bit_to_flip] = 1 - individual[bit_to_flip]
        
        # Sort the population based on fitness
        sorted_population = population[fitness.argsort()[::-1]]
        
        # Create the next generation
        new_population = np.copy(sorted_population[:elite_size])
        while len(new_population) < population_size:
            # Select parents
            parents = sorted_population[:2]
            # Perform crossover
            if random.random() < crossover_rate:
                child = np.random.choice(parents[0], n, p=parents[1]/np.sum(parents[1]))
                new_population = np.append(new_population, child, axis=0)
            # Apply mutation
            else:
                mutation_index = random.randint(0, n-1)
                new_population = np.append(new_population, np.array([population[mutation_index]]), axis=0)
        
        # Replace the old population with the new one
        population = new_population
        
    # Return the best individual from the final population
    return sorted_population[0]
