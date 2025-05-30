import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the sum of demands divided by the total capacity to normalize
    demand_ratios = demands / total_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Loop through all edges
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the heuristic value based on the distance and demand ratio
                # Negative values for undesirable edges, positive values for promising ones
                heuristics[i, j] = -distance_matrix[i, j] * demand_ratios[i]
    
    return heuristics