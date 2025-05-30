import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the heuristics are based on the ratio of prize to weight for each item
    # The dimension of weights is fixed to 1, so it simplifies to prize / weight
    # where weight is a column vector of shape (n, 1) due to the fixed dimension constraint
    # Note: This is a simple heuristic, in practice, you might want to use more sophisticated heuristics
    
    # Check if the weight matrix has the correct shape (n, 1)
    if weight.shape[1] != 1:
        raise ValueError("Weight matrix must have a shape of (n, 1)")
    
    # Calculate the heuristic values (prize to weight ratio)
    heuristics = prize / weight
    
    return heuristics