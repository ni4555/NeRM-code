import numpy as np
import numpy as np
from scipy.optimize import differential_evolution

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Define the bounds for the differential evolution algorithm
    bounds = [(0, 1) for _ in range(weight.shape[0])]
    
    # Define the objective function for the differential evolution algorithm
    def objective_function(x):
        # Calculate the total prize based on the selected items
        total_prize = np.dot(prize, np.where(x >= 0.5, 1, 0))
        
        # Calculate the total weight based on the selected items
        total_weight = np.sum(weight * np.where(x >= 0.5, 1, 0), axis=1)
        
        # Calculate the penalty for exceeding the weight constraint
        penalty = np.sum(np.where(total_weight > 1, 1, 0))
        
        # Return the total prize minus the penalty
        return total_prize - penalty
    
    # Perform differential evolution to find a near-optimal solution
    result = differential_evolution(objective_function, bounds)
    
    # Map the continuous values to binary values (0 or 1) to represent the items selected
    heuristics = np.where(result.x >= 0.5, 1, 0)
    
    return heuristics