import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the inverse of the distance matrix
    inv_distance_matrix = 1 / (distance_matrix + 1e-8)  # Add a small constant to avoid division by zero
    
    # Normalize the inverse distance matrix by the demand to get the initial heuristic
    normalized_inv_distance_matrix = inv_distance_matrix * (1 / demands)
    
    # Adjust the heuristic values based on the inverse of the demand (heuristic range adjustment)
    # Negative values indicate less promising edges, while positive values indicate more promising ones
    heuristic_matrix = -normalized_inv_distance_matrix * (1 / demands)
    
    # Sum up the heuristics along the diagonal (excluding the depot itself)
    # The sum represents the potential total distance if we visit all customers
    sum_heuristics = torch.sum(heuristic_matrix[1:], dim=0)
    
    # Normalize the heuristic matrix by the sum to get the relative importance of each edge
    relative_heuristic_matrix = heuristic_matrix / sum_heuristics
    
    return relative_heuristic_matrix