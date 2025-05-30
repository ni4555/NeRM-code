import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand at each node
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the potential for each edge to be included in a solution
    # We use a simple heuristic based on the cumulative demand at the destination node
    # We assume that a larger demand at the destination node makes the edge more promising
    # This is a simplified heuristic and can be replaced with more complex ones
    heuristics = cumulative_demand - distance_matrix
    
    return heuristics