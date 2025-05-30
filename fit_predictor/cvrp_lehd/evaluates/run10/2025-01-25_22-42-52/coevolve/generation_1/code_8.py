import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands between each pair of nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Normalize the demand difference by the vehicle capacity (1 in this case)
    normalized_demand_diff = demand_diff / demands.unsqueeze(1)
    
    # Calculate the heuristic value as the negative of the absolute demand difference
    # This encourages selecting edges with smaller absolute demand differences
    heuristics = -torch.abs(normalized_demand_diff)
    
    return heuristics