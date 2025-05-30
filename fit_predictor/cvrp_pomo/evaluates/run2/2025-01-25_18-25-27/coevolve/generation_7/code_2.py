import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands to have a common level
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Compute the heuristics values
    # Here we use a simple heuristic that takes into account both distance and demand
    # We assume that the higher the demand, the less desirable the edge, hence negative heuristic
    heuristics = -distance_matrix + normalized_demands
    
    return heuristics