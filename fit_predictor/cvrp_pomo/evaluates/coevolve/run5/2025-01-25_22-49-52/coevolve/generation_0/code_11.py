import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demand between each pair of nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Create a mask for the edges where the demand difference is positive
    positive_demand_diff_mask = (demand_diff > 0).to(torch.float32)
    
    # Calculate the heuristic value for each edge as the negative of the distance
    # multiplied by the demand difference
    heuristic_values = -distance_matrix * positive_demand_diff_mask
    
    return heuristic_values