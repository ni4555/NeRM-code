import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristic scores to be equal to the prize value of each item
    heuristics = prize.copy()

    # Calculate the sum of weights for each knapsack
    knapsack_capacities = np.sum(weight, axis=1)

    # Normalize the prize by the knapsack capacities
    normalized_prize = prize / knapsack_capacities

    # Calculate the density of each item
    density = normalized_prize / weight

    # Initialize the selection vector
    selection = np.zeros(n, dtype=bool)

    # Iteratively select items with the highest density until all knapsacks are full or all items are selected
    for _ in range(n):
        # Find the item with the maximum density
        max_density_idx = np.argmax(density[~selection])
        
        # Check if adding the item to any knapsack exceeds its capacity
        can_add = np.all(weight[max_density_idx] <= knapsack_capacities[selection], axis=1)
        
        # If it can be added to at least one knapsack, add it to the first knapsack that can accommodate it
        if np.any(can_add):
            knapsack_idx = np.where(can_add)[0][0]
            selection[knapsack_idx] = True
        else:
            break

        # Update the heuristics to indicate that the item is included in the solution
        heuristics[max_density_idx] = -np.inf  # Set to -inf to indicate item is selected

    # Return the heuristic scores as the probability of selecting each item
    return heuristics / np.sum(heuristics)
