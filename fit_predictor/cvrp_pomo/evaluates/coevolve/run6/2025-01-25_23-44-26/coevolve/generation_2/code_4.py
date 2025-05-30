import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize node demands
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic value for each edge
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                # Calculate the heuristic value as the demand at node j
                # multiplied by the total capacity divided by the demand at node i
                heuristics_matrix[i, j] = normalized_demands[j] * (total_capacity / normalized_demands[i])
    
    return heuristics_matrix