import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    heuristic_scores = np.zeros(n)
    
    # Adaptive heuristic framework to evaluate item combinations
    for i in range(n):
        # Calculate the score for each item
        item_score = np.sum(prize[i] / np.sum(weight[i]))
        heuristic_scores[i] = item_score
    
    # Robust stochastic sampling to explore diverse solution spaces
    random.shuffle(heuristic_scores)
    
    # Balance between exploration and exploitation
    adjusted_scores = heuristic_scores / np.sum(heuristic_scores)
    
    # Compliance mechanism to adhere to weight constraints
    weight_compliance = np.sum(weight, axis=1) <= 1
    
    # Dynamic adjustment of search algorithm
    for i in range(n):
        if adjusted_scores[i] > np.random.rand():
            if weight_compliance[i]:
                # Select the item if it is a promising candidate and complies with weight constraints
                continue
            else:
                # If the item does not comply, adjust its score downwards
                adjusted_scores[i] *= 0.9
    
    # Return the heuristic scores
    return adjusted_scores
