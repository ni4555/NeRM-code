import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the normalized demands
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristic based on the normalized demand of the customer
                # and the distance between the current node and the customer
                heuristics[i, j] = normalized_demands[j] * distance_matrix[i, j]
    
    return heuristics