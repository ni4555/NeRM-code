import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with a random exploration vector
    exploration = np.random.rand(weight.shape[0], 1)
    exploration /= np.linalg.norm(exploration)
    
    # Initialize the population for adaptive sampling
    population_size = int(weight.shape[0] / 2)
    population = np.random.choice(weight.shape[0], population_size, replace=False)
    
    # Main heuristic loop
    while True:
        # Evaluate the fitness of the current population
        fitness = -np.dot(prize[population], weight[population, :, 0])
        
        # Adapt the exploration vector based on fitness and random factors
        for i in range(population.shape[0]):
            # Apply robust perturbation
            mutation_factor = np.random.normal(0, 0.01)
            exploration[population[i]] += mutation_factor
        
        # Normalize the exploration vector
        exploration /= np.linalg.norm(exploration)
        
        # Select new individuals for the population using the exploration vector
        probabilities = exploration * (-fitness)
        cumulative_probabilities = np.cumsum(probabilities)
        thresholds = np.random.rand(population_size)
        new_population_indices = np.searchsorted(cumulative_probabilities, thresholds)
        new_population = np.random.choice(weight.shape[0], new_population_indices, replace=False)
        
        # Break if a termination condition is met, such as convergence or a time limit
        if np.all(fitness > 0):
            break
        
        # Replace the old population with the new one
        population = new_population
    
    # Map the exploration heuristics to the original item indices
    heuristics = np.zeros_like(prize)
    heuristics[population] = -probabilities
    heuristics = np.argmax(weight * heuristics, axis=1) * np.ones_like(prize)
    
    return heuristics
