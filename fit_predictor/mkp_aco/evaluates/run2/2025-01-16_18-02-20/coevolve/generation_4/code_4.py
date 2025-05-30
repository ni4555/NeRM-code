import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratio to sum to 1 for stochastic sampling
    total_ratio = np.sum(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / total_ratio
    
    # Generate a random number for each item and use it to rank items adaptively
    random_numbers = np.random.rand(prize.shape[0])
    
    # Rank items based on their normalized value-to-weight ratio and random number
    item_ranking = np.argsort(-normalized_ratio * random_numbers)
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros(prize.shape)
    
    # Assign a high heuristic value to the top items
    heuristics[item_ranking] = 1
    
    return heuristics