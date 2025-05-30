import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios to ensure they are in a comparable range
    normalized_ratio = (value_to_weight_ratio - np.min(value_to_weight_ratio)) / np.ptp(value_to_weight_ratio)
    
    # Sample the top items adaptively based on the normalized ratio
    # Here we use a simple random sampling, but in a real scenario, a more sophisticated method might be used
    top_items = np.argsort(normalized_ratio)[-int(0.2 * len(normalized_ratio)):]

    # Calculate the heuristics for the top items
    heuristics = np.zeros_like(prize)
    heuristics[top_items] = normalized_ratio[top_items]

    return heuristics