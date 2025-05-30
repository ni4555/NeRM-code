import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands by total capacity
    normalized_demands = demands / total_demand
    
    # Calculate the potential of each edge based on normalized demands
    # Here we use a simple heuristic: the higher the demand, the more promising the edge
    # We can adjust the weights and the heuristic function based on the problem specifics
    heuristics = normalized_demands * distance_matrix
    
    # Apply a small penalty for edges that are too far away to be considered
    # This is a simple way to prevent the algorithm from considering distant edges
    # as promising, which might lead to inefficient solutions
    penalty = 0.1 * (distance_matrix > 10).float() * distance_matrix
    heuristics -= penalty
    
    return heuristics