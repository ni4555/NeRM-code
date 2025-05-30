import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the maximum distance to ensure the scale is consistent
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Normalize the demands by the total vehicle capacity to ensure load balancing
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance heuristics
    inverse_distance_heuristics = 1 / normalized_distance_matrix
    
    # Normalize the inverse distance heuristics to ensure they are on the same scale as the demands
    normalized_inverse_distance_heuristics = inverse_distance_heuristics / torch.max(inverse_distance_heuristics)
    
    # Combine the inverse distance heuristics with the normalized demands
    combined_heuristics = normalized_inverse_distance_heuristics * normalized_demands
    
    # Subtract the combined heuristics from 1 to get negative values for undesirable edges
    heuristics = 1 - combined_heuristics
    
    return heuristics