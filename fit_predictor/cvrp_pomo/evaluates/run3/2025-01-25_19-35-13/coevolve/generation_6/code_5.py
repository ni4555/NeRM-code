import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand sum
    total_demand = torch.sum(demands)
    
    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Initialize a tensor of zeros with the same shape as the distance matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the potential heuristics based on normalized demands
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristics value based on the normalized demand
                heuristics[i, j] = normalized_demands[i] * distance_matrix[i, j]
    
    return heuristics