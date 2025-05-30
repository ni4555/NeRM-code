import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Initialize a tensor with zeros to store the heuristic values
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # Calculate the potential heuristic value
                heuristics[i, j] = distance_matrix[i, j] * normalized_demands[j]
    
    return heuristics