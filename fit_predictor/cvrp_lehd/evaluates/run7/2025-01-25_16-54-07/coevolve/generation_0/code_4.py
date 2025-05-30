import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand for each edge (distance from i to j)
    edge_demand = (demands[:-1] + demands[1:]) / distance_matrix.size(0)
    
    # Calculate the remaining capacity of each vehicle for each edge
    remaining_capacity = 1.0 - (demands[:-1] + demands[1:]) / demands.sum()
    
    # Use a simple heuristic: the more remaining capacity, the better the edge
    # Negative values for undesirable edges (demand too high), positive values for promising ones
    heuristics = -edge_demand + remaining_capacity
    
    return heuristics