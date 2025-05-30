import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check if the demands are normalized
    if demands.sum() != 1:
        raise ValueError("Demands must be normalized by the total vehicle capacity.")
    
    # Calculate the sum of demands (should be 1 if demands are normalized)
    sum_demands = demands.sum()
    
    # Calculate the difference in demands between each pair of nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the heuristic value as the negative of the difference in demands
    # multiplied by the distance between nodes
    heuristics = -torch.abs(demand_diff) * distance_matrix
    
    # Normalize the heuristics by the sum of demands to maintain the same scale
    heuristics /= sum_demands
    
    return heuristics