import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand difference matrix
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Avoid negative demand differences, which would indicate an undesirable edge
    demand_diff = torch.clamp(demand_diff, min=0)
    
    # Calculate the distance penalty matrix (the larger the distance, the higher the penalty)
    distance_penalty = distance_matrix * (1 + demand_diff)
    
    # Normalize by the maximum distance to prevent overflow
    max_distance = distance_matrix.max()
    distance_penalty = distance_penalty / max_distance
    
    # Return the heuristics matrix
    return distance_penalty