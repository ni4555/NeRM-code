import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the sum of demands and distance squared for each edge
    edge_values = (distance_matrix**2) * normalized_demands
    
    # The heuristic value is the sum of the edge value and the inverse of the distance
    # to encourage the selection of shorter edges first. We subtract this from 1
    # to have negative values for undesirable edges and positive for promising ones.
    heuristics = 1 - (edge_values + torch.inverse(distance_matrix))
    
    return heuristics