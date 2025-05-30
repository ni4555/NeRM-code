import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate cumulative demand mask
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Normalize cumulative demand
    normalized_demand = cumulative_demand / demands.sum()
    
    # Calculate the cumulative demand difference from the total capacity
    capacity_difference = cumulative_demand - demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # For each edge (i, j), calculate the heuristic value
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Skip the depot node
                # Calculate the edge heuristic based on normalized demand and capacity difference
                heuristics[i, j] = normalized_demand[j] - capacity_difference[j]
    
    return heuristics