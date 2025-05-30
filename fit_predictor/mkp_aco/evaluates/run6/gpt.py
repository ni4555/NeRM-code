import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    value_to_weight_ratio = prize / np.sum(weight, axis=1)
    # Apply a non-linear transformation by squaring the ratio and emphasizing high values
    non_linear_scaled_ratio = (value_to_weight_ratio ** 2) * (value_to_weight_ratio > 0.5)
    # Incorporate diversity by penalizing low-value to weight ratios using exponential decay
    diversity_factor = np.exp(-value_to_weight_ratio * (value_to_weight_ratio < 0.1))
    # Combine the non-linear scaled ratio with the diversity factor
    combined_heuristics = non_linear_scaled_ratio * diversity_factor
    # Apply a sparsification technique by setting values below a certain threshold to zero
    sparsified_heuristics = np.where(combined_heuristics > 0.1, combined_heuristics, 0)
    # Normalize the sparsified heuristics to ensure they sum to 1
    heuristics = sparsified_heuristics / np.sum(sparsified_heuristics)
    return heuristics
