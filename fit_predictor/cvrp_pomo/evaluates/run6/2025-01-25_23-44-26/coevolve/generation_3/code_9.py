import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of customer demands
    vehicle_capacity = demands.sum()
    
    # Normalize the demands vector by the vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Normalize the distance matrix
    min_distance = distance_matrix.min()
    max_distance = distance_matrix.max()
    normalized_distance_matrix = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Heuristic calculation: the heuristic value for an edge (i, j) is the product of the normalized distance
    # and the normalized demand. A negative value indicates an undesirable edge, and a positive value indicates
    # a promising edge.
    heuristics = -normalized_distance_matrix * normalized_demands
    
    return heuristics