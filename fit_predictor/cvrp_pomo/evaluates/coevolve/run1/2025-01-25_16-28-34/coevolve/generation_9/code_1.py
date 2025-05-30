import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize the demands to be between 0 and 1
    demands = demands / demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate over each edge
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip the diagonal
                # Calculate the heuristic value based on the distance and the demand ratio
                heuristics[i, j] = distance_matrix[i, j] * (demands[j] - demands[i])
    
    return heuristics