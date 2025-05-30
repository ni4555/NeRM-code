import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    demand_normalized = demands / vehicle_capacity
    distance_squared = distance_matrix ** 2
    
    # Calculate heuristic values based on demand and distance
    heuristics = -distance_squared + demand_normalized.unsqueeze(1) * distance_matrix
    
    return heuristics