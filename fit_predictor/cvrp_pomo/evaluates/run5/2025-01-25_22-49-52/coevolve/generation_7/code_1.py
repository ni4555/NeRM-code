import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands by the total capacity
    normalized_demands = demands / total_capacity
    
    # Normalize the distance matrix by the maximum distance in the matrix
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate the potential value for each edge
    # Here we use a simple heuristic: potential = -distance + demand
    # This heuristic prioritizes shorter distances and higher demands
    potential_matrix = -normalized_distance_matrix + normalized_demands
    
    # Ensure that the potential values are within the range [-1, 0] for undesirable edges
    # and [0, 1] for promising ones by adding a small constant to the minimum potential
    # and subtracting it from the maximum potential
    min_potential = torch.min(potential_matrix)
    max_potential = torch.max(potential_matrix)
    potential_matrix = (potential_matrix - min_potential) / (max_potential - min_potential)
    
    return potential_matrix