import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance heuristic
    inv_distance_heuristic = -distance_matrix
    
    # Calculate the demand normalization heuristic
    demand_normalization_heuristic = normalized_demands.expand_as(distance_matrix)
    
    # Combine both heuristics
    combined_heuristic = inv_distance_heuristic + demand_normalization_heuristic
    
    return combined_heuristic