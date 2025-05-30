import random
import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]

    # Calculate the total weight for each item
    item_weight = np.sum(weight, axis=1)
    # Calculate the ratio of prize to weight for each item
    ratio = prize / item_weight
    # Normalize the ratio to get the initial heuristics
    max_ratio = np.max(ratio)
    heuristics = ratio / max_ratio
    
    # Integrate additional factors such as inverse of item dimensionality
    dimensionality = np.prod(weight, axis=1)
    additional_factor = 1 / (dimensionality + 1)  # Adding 1 to avoid division by zero
    
    # Combine factors to create a new heuristic
    combined_heuristics = heuristics * additional_factor
    
    # Calculate the variance of the combined heuristics
    variance = np.var(combined_heuristics)
    # Dynamically adapt mutation rate based on variance
    mutation_rate = np.sqrt(variance) / np.max(combined_heuristics)
    
    # Mutate the heuristics to enhance diversity
    mutation_indices = np.random.choice(len(combined_heuristics), size=int(len(combined_heuristics) * mutation_rate), replace=False)
    noise = np.random.normal(0, np.std(combined_heuristics), len(mutation_indices))  # Noise scaled by standard deviation
    combined_heuristics[mutation_indices] += noise
    
    # Normalize the combined heuristics to ensure they sum to 1 for easier comparison
    combined_heuristics /= np.sum(combined_heuristics)
    
    # Use adaptive thresholds based on the percentiles of the combined heuristics
    threshold_low = np.percentile(combined_heuristics, 10)  # Lower threshold at 10th percentile
    threshold_high = np.percentile(combined_heuristics, 90)  # Higher threshold at 90th percentile
    
    # Sparsify the heuristics by setting values below the lower threshold to zero and values above the higher threshold to 1
    sparsified_heuristics = np.where(combined_heuristics < threshold_low, 0,
                                    np.where(combined_heuristics > threshold_high, 1, combined_heuristics))
    
    # Multi-start optimization by re-running the heuristic calculation multiple times
    best_heuristics = sparsified_heuristics.copy()
    for _ in range(3):  # Reduced number of restarts
        # Re-calculate heuristics with the current best as a starting point
        combined_heuristics = heuristics * additional_factor
        combined_heuristics /= np.sum(combined_heuristics)
        sparsified_heuristics = np.where(combined_heuristics < threshold_low, 0,
                                        np.where(combined_heuristics > threshold_high, 1, combined_heuristics))
        best_heuristics = np.maximum(best_heuristics, sparsified_heuristics)
    
    return best_heuristics
