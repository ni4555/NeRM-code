import numpy as np
import numpy as np
from scipy.optimize import differential_evolution

def heuristic_fitness(individual, prize, weight, bounds):
    # Calculate the total prize for the selected items
    selected_items = individual.astype(bool)
    total_prize = np.sum(prize[selected_items])
    
    # Calculate the total weight for the selected items
    total_weight = np.sum(weight[selected_items, :], axis=1)
    
    # Check if the total weight is within the constraints
    if np.any(total_weight > 1):
        return 0  # Constraint violated
    
    return total_prize

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Define bounds for the differential evolution algorithm
    bounds = [(0, 1) for _ in range(weight.shape[0])]
    
    # Define the differential evolution strategy
    strategy = {'mutation': (0.5, 1), 'recombination': 0.7, 'crosspb': 0.5}
    
    # Run differential evolution to find the optimal subset of items
    result = differential_evolution(
        lambda individual: -heuristic_fitness(individual, prize, weight, bounds),  # Minimize the negative of the heuristic to maximize the prize
        bounds,
        strategy=strategy,
        seed=42
    )
    
    # Convert the result to a binary array where 1 indicates selected and 0 indicates not selected
    heuristics = np.zeros_like(prize, dtype=float)
    heuristics[result.x] = 1
    
    return heuristics