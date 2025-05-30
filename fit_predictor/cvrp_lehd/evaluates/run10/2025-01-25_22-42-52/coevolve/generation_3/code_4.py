import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum distance in the matrix to use as a scaling factor
    max_distance = torch.max(distance_matrix)
    
    # Normalize the distance matrix with the maximum distance to scale the values
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Normalize the demands by the total vehicle capacity
    vehicle_capacity = demands.sum()
    normalized_demands = demands / vehicle_capacity
    
    # Compute the heuristics as the product of the normalized distance and the normalized demand
    heuristics = normalized_distance_matrix * normalized_demands
    
    # To ensure the matrix contains negative values for undesirable edges and positive ones for promising ones,
    # we add a constant that is the sum of the maximum distance and the maximum demand.
    # This constant ensures that at least one edge is considered promising (has a positive heuristic value).
    constant = max_distance + torch.max(normalized_demands)
    heuristics = heuristics + constant
    
    return heuristics