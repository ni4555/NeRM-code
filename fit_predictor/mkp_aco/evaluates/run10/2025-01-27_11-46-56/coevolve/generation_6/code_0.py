import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape
    assert weight.shape == (n, m), "The shape of prize and weight must match."
    
    # Define a helper function for Genetic Algorithm's selection
    def selection(population, probabilities):
        cumulative_probabilities = np.cumsum(probabilities)
        random_value = np.random.rand()
        for i, probability in enumerate(cumulative_probabilities):
            if random_value < probability:
                return i
    
    # Initialize GA population
    population_size = 50
    population = np.random.choice([0, 1], size=(population_size, n), p=np.exp(-prize/100))
    
    # Genetic Algorithm parameters
    crossover_rate = 0.8
    mutation_rate = 0.02
    elite_count = 1
    
    # Initialize SA parameters
    initial_temperature = 10000
    final_temperature = 1
    cooling_rate = 0.99
    
    # GA and SA combined algorithm
    best_population = population
    best_fitness = np.sum(best_population * prize)
    temperature = initial_temperature
    
    while temperature > final_temperature:
        # Simulated Annealing step
        current_population = population.copy()
        for _ in range(int(mutation_rate * population_size)):
            idx = np.random.randint(n)
            if current_population[idx] == 1:
                current_population[idx] = 0
            else:
                current_population[idx] = 1
        
        # Genetic Algorithm step
        new_population = np.zeros_like(population)
        elite_indices = np.argsort(np.sum(population * prize, axis=1))[-elite_count:]
        elite_fitness = np.sum(population[elite_indices] * prize, axis=1)
        non_elite_indices = [i for i in range(population_size) if i not in elite_indices]
        for i in non_elite_indices:
            # Selection
            parent1_index = selection(population, np.exp((elite_fitness - np.sum(population * prize, axis=1)) / temperature))
            parent2_index = selection(population, np.exp((elite_fitness - np.sum(population * prize, axis=1)) / temperature))
            
            # Crossover
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, n)
                new_population[i, :crossover_point] = population[parent1_index, :crossover_point]
                new_population[i, crossover_point:] = population[parent2_index, crossover_point:]
            else:
                new_population[i] = population[parent1_index]
        
        new_population[elite_indices] = population[elite_indices]
        new_population = np.clip(new_population, 0, 1)
        
        # Replace the old population if the new one is better
        new_fitness = np.sum(new_population * prize)
        if new_fitness > best_fitness:
            best_population = new_population
            best_fitness = new_fitness
        
        # Cooling schedule for temperature
        temperature *= cooling_rate
    
    # Heuristics
    heuristics = np.exp((prize - best_population * prize) / best_fitness) / np.exp((prize - best_population * prize) / best_fitness).sum()
    
    return heuristics
