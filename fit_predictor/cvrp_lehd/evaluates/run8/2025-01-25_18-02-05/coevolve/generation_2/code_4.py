import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the sum of all demands
    demands_sum = demands.sum()
    normalized_demands = demands / demands_sum
    
    # Calculate the cost of each edge by taking the distance
    edge_costs = distance_matrix
    
    # Calculate the negative sum of demands and distance for each edge
    # The more negative the value, the more promising the edge is to be included
    heuristics = -edge_costs * normalized_demands
    
    return heuristics