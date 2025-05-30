import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Normalize the demands
    total_capacity = torch.sum(demands)
    normalized_demands = demands / total_capacity
    
    # Compute potential value for each edge
    # The potential value is a combination of normalized distance and normalized demand
    # Negative values for undesirable edges, positive values for promising ones
    potential_matrix = normalized_distance_matrix * normalized_demands
    
    # Subtract the potential value from the maximum possible value to get negative values
    # for undesirable edges
    max_potential = torch.max(potential_matrix)
    heuristics = max_potential - potential_matrix
    
    return heuristics