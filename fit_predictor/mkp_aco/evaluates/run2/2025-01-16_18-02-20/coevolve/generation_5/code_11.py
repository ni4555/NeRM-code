import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio by summing across dimensions
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum(axis=1)[:, np.newaxis]
    
    # Rank items based on their normalized value-to-weight ratio in descending order
    ranking = np.argsort(-normalized_ratio, axis=1)
    
    # Calculate the heuristic score for each item based on its ranking
    heuristics = np.zeros_like(prize)
    for i in range(n):
        # Get the index of the item with the highest normalized ratio
        index = ranking[i][0]
        # Calculate the heuristic score for the item
        heuristics[i] = normalized_ratio[i][index]
    
    return heuristics