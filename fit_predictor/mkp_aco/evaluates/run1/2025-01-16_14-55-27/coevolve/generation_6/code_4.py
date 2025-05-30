import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Calculate the total prize for each dimension
    total_prize = prize.sum(axis=0)
    
    # Dynamic item sorting based on weighted ratio
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Apply weighted ratio analysis to compute the heuristics
    heuristics = np.zeros_like(prize)
    for i in sorted_indices:
        # For each item, calculate the potential prize contribution
        potential_prize = 0
        for j in range(m):
            # If adding this item does not exceed the weight constraint, add its contribution
            if potential_prize + prize[i] <= total_prize[j]:
                potential_prize += prize[i]
            else:
                break
        
        # The heuristic value is the ratio of the potential prize to the total prize
        heuristics[i] = potential_prize / total_prize[i]
    
    return heuristics