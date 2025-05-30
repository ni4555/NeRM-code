import numpy as np
import numpy as np
from sklearn.preprocessing import normalize

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    
    # Sample items randomly and calculate their heuristic scores
    random_indices = np.random.choice(n, size=int(0.1 * n), replace=False)
    random_heuristics = np.abs(prize[random_indices] - weight[random_indices].sum(axis=1))
    
    # Use weighted ratio analysis for all items
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Combine random heuristic samples with weighted ratio analysis
    combined_heuristics = np.abs(weighted_ratio - 1) * 0.5 + np.abs(random_heuristics) * 0.5
    
    # Normalize the heuristics to make sure they are between 0 and 1
    heuristics = normalize(combined_heuristics, axis=0)
    
    return heuristics