import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total value per unit weight for each item in each dimension
    value_per_weight = prize / weight
    
    # Normalize the value per weight across dimensions to create a heuristic
    # that takes into account the overall value contribution of each item
    max_value_per_weight = np.max(value_per_weight, axis=1)
    normalized_value_per_weight = value_per_weight / max_value_per_weight[:, np.newaxis]
    
    # Sum the normalized value per weight across dimensions to get a final heuristic
    heuristics = np.sum(normalized_value_per_weight, axis=1)
    
    return heuristics