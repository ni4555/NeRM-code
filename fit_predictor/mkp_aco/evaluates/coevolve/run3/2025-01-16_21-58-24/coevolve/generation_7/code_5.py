import numpy as np
import numpy as np
from scipy.optimize import differential_evolution

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Define the objective function for the differential evolution algorithm
    def objective_function(x):
        # Calculate the total prize for the selected items
        selected_prize = np.sum(prize[x > 0])
        # Calculate the total weight for the selected items
        selected_weight = np.sum(weight[x > 0], axis=1)
        # Check if the total weight is within the constraints
        if np.all(selected_weight <= 1):
            return -selected_prize  # Maximize the negative prize to minimize the function
        else:
            return -np.inf  # Return negative infinity if constraints are violated

    # Initialize the bounds for each item, where 0 means the item is not selected and 1 means it is selected
    bounds = [(0, 1) for _ in range(weight.shape[0])]

    # Perform the differential evolution to find the optimal subset of items
    result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=1000, popsize=50)

    # Convert the binary result to a heuristic score
    heuristics = np.zeros_like(prize)
    heuristics[result.x > 0] = 1

    return heuristics