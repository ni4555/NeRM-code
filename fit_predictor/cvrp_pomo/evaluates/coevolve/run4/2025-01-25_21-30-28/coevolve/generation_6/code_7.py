import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized inverse distance heuristics
    # This is a simple heuristic that assumes that closer customers are more likely to be visited first
    # and are therefore more promising. Negative values are used to indicate undesirable edges.
    inverse_distance_heuristic = 1.0 / (distance_matrix + 1e-8)  # Add a small value to avoid division by zero
    
    # Normalize the inverse distance heuristics by the demand to account for load balancing
    # This heuristic assumes that customers with smaller demands are more promising due to easier load management
    normalized_inverse_distance_heuristic = inverse_distance_heuristic * (1.0 / (demands + 1e-8))
    
    # Normalize the entire matrix so that the sum of heuristics for each row (customer) is equal to 1
    # This step normalizes the matrix by the sum of each row, ensuring that the heuristics are relative
    # to each other and not the absolute values.
    row_sums = normalized_inverse_distance_heuristic.sum(dim=1, keepdim=True)
    normalized_heuristics = normalized_inverse_distance_heuristic / row_sums
    
    return normalized_heuristics