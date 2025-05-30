import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance
    inv_distance = 1 / distance_matrix
    
    # Normalize the inverse distance matrix
    max_distance = torch.max(inv_distance, dim=1, keepdim=True)[0]
    normalized_inv_distance = inv_distance / max_distance
    
    # Normalize the demands
    total_capacity = torch.sum(demands)
    normalized_demands = demands / total_capacity
    
    # Combine the normalized inverse distance and normalized demands
    combined_heuristic = normalized_inv_distance - normalized_demands
    
    return combined_heuristic