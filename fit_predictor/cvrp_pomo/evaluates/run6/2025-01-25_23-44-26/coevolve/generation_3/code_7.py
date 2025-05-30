import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the demand-to-distance ratio for each edge
    demand_to_distance_ratio = (demands.unsqueeze(1) / total_capacity) * distance_matrix
    
    # Normalize the ratio to have a range between 0 and 1
    demand_to_distance_ratio = (demand_to_distance_ratio - demand_to_distance_ratio.min()) / (demand_to_distance_ratio.max() - demand_to_distance_ratio.min())
    
    # Apply a penalty for high demand-to-distance ratios (undesirable edges)
    # Here, a simple linear penalty is used, but this can be replaced with a more sophisticated function
    penalty = (1 - demand_to_distance_ratio) * 100  # Negative values indicate undesirable edges
    
    return penalty