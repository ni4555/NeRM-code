import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to have a sum of 1 for the heuristic calculation
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum
    
    # Compute the heuristic value for each edge
    # Heuristic value is the negative of the ratio of the distance and demand
    # This encourages paths with lower distances and lower demands
    heuristics = -distance_matrix * normalized_demands
    
    return heuristics