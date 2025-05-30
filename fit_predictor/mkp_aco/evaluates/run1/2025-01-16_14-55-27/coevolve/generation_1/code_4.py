import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Since each dimension's constraint is fixed to 1, we can sum the weights across dimensions
    # to determine the total weight of each item.
    total_weight = np.sum(weight, axis=1)
    
    # Assuming that the prize is proportional to the "promise" of an item,
    # we can use the ratio of prize to weight as a heuristic score.
    # However, if we want to optimize for both prize collection and weight constraint management,
    # we can normalize this ratio to the maximum possible ratio to get a relative heuristic score.
    max_ratio = np.max(prize / total_weight)
    heuristic_scores = prize / total_weight / max_ratio
    
    # The resulting heuristic score for each item is a measure of how promising it is to include
    # the item in the solution, considering the prize and weight.
    return heuristic_scores