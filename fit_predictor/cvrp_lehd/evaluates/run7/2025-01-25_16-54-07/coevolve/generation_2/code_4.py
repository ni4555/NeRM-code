import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum distance in the matrix to use as a reference for normalization
    max_distance = torch.max(distance_matrix)
    
    # Normalize the distance matrix based on the maximum distance
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate the sum of normalized demands to use for further normalization
    sum_normalized_demands = torch.sum(demands)
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / sum_normalized_demands
    
    # Calculate the potential utility of each edge as a combination of distance and demand
    # Here we use a simple heuristic that gives higher utility to edges with lower distance
    # and lower normalized demand.
    # Negative values are used for undesirable edges, and positive values for promising ones.
    utility_matrix = -normalized_distance_matrix + normalized_demands
    
    # Optionally, you can add more sophisticated heuristics here
    # For example, you could consider vehicle capacity constraints by adding a term
    # that penalizes edges that would violate the capacity constraint, but this
    # depends on the specifics of the problem.

    return utility_matrix