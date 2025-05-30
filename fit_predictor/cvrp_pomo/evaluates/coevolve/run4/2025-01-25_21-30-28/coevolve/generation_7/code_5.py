import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Inverse distance heuristic: closer nodes have higher priority
    inv_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero
    
    # Demand normalization heuristic: balance the allocation of customer demands
    demand_normalized = normalized_demands * inv_distance
    
    # Combine heuristics
    heuristics = -demand_normalized
    
    return heuristics