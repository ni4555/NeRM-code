import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # Calculate the heuristic value based on the normalized demand and distance
                heuristics[i, j] = normalized_demands[i] * distance_matrix[i, j]
            else:
                # No cost for the depot node
                heuristics[i, j] = 0
    
    return heuristics