import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values based on normalized demands and distance
    # Here we use a simple heuristic: the more distant the edge, the less promising it is
    # and the more demand it has, the more promising it is.
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1) * distance_matrix
    
    return heuristics