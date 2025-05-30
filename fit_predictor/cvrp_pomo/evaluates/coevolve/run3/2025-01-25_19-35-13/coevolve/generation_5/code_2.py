import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be between 0 and 1
    normalized_demands = demands / demands.sum()
    
    # Initialize the heuristic matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Skip the diagonal (no distance to self)
                # Calculate the heuristic based on normalized demand and distance
                heuristics[i, j] = -distance_matrix[i, j] * normalized_demands[i] * normalized_demands[j]
    
    return heuristics