import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the distance-demand evaluation for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Skip the diagonal (no self-loops)
                # The heuristic value is the negative of the normalized demand multiplied by the distance
                # This encourages selecting edges with lower demand first
                heuristics[i, j] = -normalized_demands[i] * distance_matrix[i, j]
    
    return heuristics