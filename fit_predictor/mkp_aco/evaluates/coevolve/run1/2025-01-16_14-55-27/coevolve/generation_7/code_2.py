import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the prize and weight are numpy arrays with proper shapes
    # and the weight constraint is fixed to 1 for each dimension.
    
    # Calculate the ratio of prize to weight for each item
    ratio = prize / weight
    
    # Normalize the ratios using a dynamic weighted ratio index
    # Here, we use a simple normalization where we normalize by the mean ratio
    # This is just an example, the actual normalization method should be defined
    # based on the problem requirements and constraints.
    normalized_ratio = ratio / np.mean(ratio)
    
    # Apply adaptive probabilistic sampling
    # We sample items based on their normalized ratio, the probability of selecting
    # an item is proportional to its normalized ratio.
    # Here, we use a simple sampling method where we multiply by a random variable
    # between 0 and 1 to simulate this process.
    random_sample = np.random.rand(len(ratio))
    probabilities = normalized_ratio * random_sample
    
    # Proactive item selection
    # We select items based on the probabilities obtained from the sampling process
    selected_items = np.where(probabilities > np.random.rand(len(probabilities)))[0]
    
    # Create the heuristics array where each item's score is 1 if selected, otherwise 0
    heuristics = np.zeros(len(prize))
    heuristics[selected_items] = 1
    
    return heuristics