import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the prize and weight are already normalized and processed as required
    # We will use a simple heuristic: the ratio of the prize to the total weight of the item
    # which is 1 since the weight of each dimension is fixed to 1
    heuristic_values = prize / np.sum(weight, axis=1)
    return heuristic_values