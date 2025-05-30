import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Normalize the demands by the total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance heuristic
    min_distance = distance_matrix.min(dim=1)[0]
    inv_distance = 1.0 / min_distance
    
    # Calculate the demand normalization heuristic
    demand_normalized = normalized_demands * distance_matrix
    
    # Combine the two heuristics
    combined_heuristics = inv_distance + demand_normalized
    
    # Clamp the values to ensure they are within a reasonable range
    combined_heuristics = torch.clamp(combined_heuristics, min=-1e9, max=1e9)
    
    return combined_heuristics