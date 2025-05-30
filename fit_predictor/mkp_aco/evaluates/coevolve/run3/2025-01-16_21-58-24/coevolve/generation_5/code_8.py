import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = len(prize)
    m = len(weight[0])
    # Initialize the heuristic scores for each item
    heuristic_scores = np.zeros(n)
    
    # Calculate the weighted sum for each item
    for i in range(n):
        weighted_sum = np.sum(prize[i] * weight[i])
    
    # Calculate the adherence to constraints (all dimensions are 1)
    adherence = np.sum(weight, axis=1) <= m
    
    # Combine the weighted sum and adherence into heuristic scores
    heuristic_scores = weighted_sum * adherence
    
    return heuristic_scores