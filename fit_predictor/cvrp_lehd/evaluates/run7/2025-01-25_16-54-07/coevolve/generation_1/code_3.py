import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    # Normalize the demands by the total capacity
    normalized_demands = demands / total_demand
    
    # Initialize the heuristic matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values based on the normalized demands
    for i in range(n):
        for j in range(n):
            # Calculate the heuristics value based on the distance and the normalized demand
            heuristics_matrix[i, j] = distance_matrix[i, j] - normalized_demands[i] * demands[j]
    
    return heuristics_matrix