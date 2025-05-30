import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the value/weight ratio heuristic is used, we calculate the ratio for each item
    # Here, the weight is treated as a 2D array (n, m) but since m=1, it is essentially a 1D array for each item
    value_weight_ratio = prize / weight
    
    # The heuristic score is simply the value/weight ratio, indicating how promising it is to include an item
    heuristics = value_weight_ratio
    
    return heuristics