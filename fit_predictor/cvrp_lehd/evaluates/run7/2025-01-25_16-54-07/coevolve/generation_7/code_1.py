import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand for each node
    cumulative_demand = demands.cumsum(dim=0)
    
    # Calculate the normalized demand for each node
    normalized_demand = cumulative_demand / demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristic value based on distance and demand
                heuristics[i, j] = -distance_matrix[i, j] - normalized_demand[i] + normalized_demand[j]
    
    return heuristics