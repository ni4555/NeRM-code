import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to create a relative distance
    normalized_distance = distance_matrix / torch.max(distance_matrix)
    
    # Normalize the demands to fit the vehicle capacity
    normalized_demands = demands / torch.sum(demands)
    
    # Inverse of the distance as a heuristic for edges
    inverse_distance = 1 / normalized_distance
    
    # Combine the heuristics: inverse distance and normalized demands
    # Demands are subtracted because we want to prioritize edges with lower demand
    combined_heuristics = inverse_distance - normalized_demands
    
    # Ensure that no edge is assigned a negative heuristic value
    combined_heuristics = torch.clamp(combined_heuristics, min=0)
    
    return combined_heuristics