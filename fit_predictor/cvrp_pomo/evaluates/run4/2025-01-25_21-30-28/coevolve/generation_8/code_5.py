import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate inverse distance
    inv_distance_matrix = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Normalize demands
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum
    
    # Calculate heuristics based on inverse distance and demand normalization
    heuristics = -inv_distance_matrix * normalized_demands
    
    return heuristics