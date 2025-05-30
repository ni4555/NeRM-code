import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the demand-based heuristic
    demand_heuristic = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Combine the normalized distance and demand-based heuristic
    combined_heuristic = normalized_distance_matrix - demand_heuristic
    
    # Ensure that the combined heuristic does not have negative values
    combined_heuristic = torch.clamp(combined_heuristic, min=0)
    
    return combined_heuristic