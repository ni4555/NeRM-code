import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand difference for each edge
    normalized_demand_diff = (demands - demands[:, None]) / demands.sum()
    
    # Calculate the negative of the distance matrix to make shorter distances more promising
    negative_distance = -distance_matrix
    
    # Combine the normalized demand difference and negative distance
    combined_heuristics = negative_distance + normalized_demand_diff
    
    # Replace negative values with zeros to indicate undesirable edges
    combined_heuristics = torch.clamp(combined_heuristics, min=0)
    
    return combined_heuristics