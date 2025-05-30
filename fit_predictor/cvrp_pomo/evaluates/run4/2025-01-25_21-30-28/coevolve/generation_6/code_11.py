import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to the range of [0, 1]
    max_demand = torch.max(demands)
    normalized_demands = demands / max_demand
    
    # Calculate the inverse of the distance matrix, which is used in the heuristic
    # Note: To avoid division by zero, we use max_distance as a fallback
    max_distance = torch.max(distance_matrix)
    inverse_distance = torch.clamp(1.0 / (distance_matrix + 1e-10), min=0)
    
    # Combine the normalized demands and inverse distance to form the heuristic
    # Negative values indicate undesirable edges
    heuristics = -normalized_demands * inverse_distance
    
    return heuristics