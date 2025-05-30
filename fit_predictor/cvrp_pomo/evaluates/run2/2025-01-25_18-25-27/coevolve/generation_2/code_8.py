import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize demands to be between 0 and 1
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic value for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the heuristic value based on the normalized demand
                heuristics[i][j] = normalized_demands[i] * normalized_demands[j] * distance_matrix[i][j]
    
    return heuristics