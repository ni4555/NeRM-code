import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the heuristic function is a simple normalization of the prize per unit weight
    # across all dimensions, which is then summed to get the overall heuristic for each item.
    # This is a naive approach and may not be effective for complex MKP instances.
    
    # Calculate the total prize per unit weight for each item across all dimensions
    total_prize_per_unit_weight = prize / weight.sum(axis=1)
    
    # Sum the total prize per unit weight across dimensions to get the heuristic value for each item
    heuristics = total_prize_per_unit_weight.sum(axis=1)
    
    return heuristics