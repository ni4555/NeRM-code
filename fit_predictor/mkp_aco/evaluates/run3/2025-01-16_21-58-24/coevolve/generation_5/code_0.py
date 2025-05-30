import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the fitness score for each item based on the prize and weight
    fitness_scores = prize / np.sum(weight, axis=1)
    
    # Calculate the penalty for each item based on the adherence to constraints
    penalty_scores = np.sum(weight, axis=1)  # Assuming the constraint is to not exceed 1 for each dimension
    
    # Combine fitness and penalty scores into a heuristic score
    heuristic_scores = fitness_scores - penalty_scores
    
    return heuristic_scores