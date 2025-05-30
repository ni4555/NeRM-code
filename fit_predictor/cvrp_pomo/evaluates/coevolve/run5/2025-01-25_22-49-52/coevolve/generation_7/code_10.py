import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by dividing each row by the maximum value in that row
    normalized_distance_matrix = distance_matrix / distance_matrix.max(dim=1, keepdim=True)[0]
    
    # Normalize the demands by dividing each demand by the total vehicle capacity
    vehicle_capacity = demands.sum()
    normalized_demands = demands / vehicle_capacity
    
    # Compute the potential value for each edge based on normalized distance and demand
    # The heuristic function: heur = -distance + demand
    heuristics = -normalized_distance_matrix + normalized_demands
    
    # Ensure that the heuristics matrix has the same shape as the distance matrix
    assert heuristics.shape == distance_matrix.shape, "Heuristics shape does not match distance matrix shape"
    
    return heuristics