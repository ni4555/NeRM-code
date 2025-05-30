import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Normalize the distance matrix (divide by the maximum distance)
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Compute the potential value for each edge
    # This is a simple example, more sophisticated heuristics can be applied
    potential_values = -normalized_distance_matrix * normalized_demands
    
    # The heuristics should return negative values for undesirable edges
    # and positive values for promising ones, we can simply use the potential values
    return potential_values