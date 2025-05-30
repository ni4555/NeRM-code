import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize the distance matrix to have a range between 0 and 1
    normalized_distance_matrix = distance_matrix / torch.max(distance_matrix)
    
    # Compute the load factor for each edge
    load_factor = 2 * demands[:, None] * demands[None, :] / (demands.sum() + 1e-6)
    
    # Define a heuristic that combines the normalized distance and load factor
    # We use a weighted sum of the normalized distance and load factor, where
    # the weight is set to emphasize the importance of load balancing over distance
    weight = 0.5
    heuristic_matrix = -weight * normalized_distance_matrix + (1 - weight) * load_factor
    
    return heuristic_matrix