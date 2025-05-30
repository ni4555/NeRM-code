import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand for each edge
    edge_cumulative_demand = demands.cumsum(0)
    
    # Calculate the cumulative distance for each edge
    edge_cumulative_distance = distance_matrix.cumsum(1)
    
    # Normalize cumulative demand and cumulative distance
    edge_cumulative_demand /= demands.sum()
    edge_cumulative_distance /= distance_matrix.sum(1)
    
    # Calculate the heuristic values as the negative of the cumulative distance
    # and add a positive bias for edges with cumulative demand close to 1
    heuristics = -edge_cumulative_distance + (1 - edge_cumulative_demand) * 1000
    
    return heuristics