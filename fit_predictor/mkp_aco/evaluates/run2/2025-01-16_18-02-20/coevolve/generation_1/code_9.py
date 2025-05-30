import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the prize is a scalar for each item
    assert prize.ndim == 1 and prize.shape[0] == weight.shape[0]
    
    # Calculate the heuristic for each item based on prize to weight ratio
    heuristics = prize / weight
    
    # You might want to add some form of normalization to the heuristic values
    # if they are not in a useful range or scale, but this is omitted here for simplicity.
    
    return heuristics