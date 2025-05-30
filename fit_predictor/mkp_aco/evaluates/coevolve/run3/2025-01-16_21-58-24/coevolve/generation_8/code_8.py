import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # This is a simple heuristic that calculates the value-to-weight ratio for each item
    # and multiplies it by a random factor to introduce some stochasticity in the heuristic
    # values. The idea is to favor items with a high value-to-weight ratio, but still
    # allow some randomness to explore the solution space more thoroughly.
    
    value_to_weight_ratio = prize / weight.sum(axis=1)
    random_factor = np.random.rand(prize.size)
    heuristics = value_to_weight_ratio * random_factor
    
    return heuristics