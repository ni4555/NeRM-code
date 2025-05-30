import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands to determine the relative importance of the edges
    cumulative_demands = torch.cumsum(demands, dim=0)
    
    # Calculate the heuristic values based on the distance and cumulative demands
    # Negative values indicate undesirable edges (e.g., edges that would cause overflow)
    heuristics = -distance_matrix + cumulative_demands
    
    return heuristics