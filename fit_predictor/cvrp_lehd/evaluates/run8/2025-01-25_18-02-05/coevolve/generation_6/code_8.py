import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands between the current node and the depot
    demand_diff = demands - demands[0]
    
    # Calculate the heuristic value as the negative of the difference in demand
    # multiplied by the distance from the current node to the depot
    heuristics = -torch.abs(demand_diff) * distance_matrix[:, 0]
    
    return heuristics