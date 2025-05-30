import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor of zeros with the same shape as distance_matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Compute the heuristics based on normalized demands
    for i in range(n):
        for j in range(n):
            # If it's the same node or the depot, assign a high heuristic value
            if i == j or i == 0:
                heuristics[i, j] = 100
            # Otherwise, assign negative values based on the demand and distance
            else:
                heuristics[i, j] = -distance_matrix[i, j] - normalized_demands[j]
    
    return heuristics