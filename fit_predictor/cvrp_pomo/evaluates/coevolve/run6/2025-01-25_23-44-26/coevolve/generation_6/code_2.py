import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Inverse Distance Heuristic (IDH): Assign customers based on the reciprocal of their distance
    idh_scores = 1.0 / distance_matrix
    
    # Demand penalty function: Increase cost for customers near vehicle capacity
    demand_penalty = (normalized_demands * distance_matrix).sum(dim=1)
    demand_penalty = demand_penalty.unsqueeze(1) * distance_matrix
    
    # Combine IDH scores with demand penalty to get initial heuristics
    combined_heuristics = idh_scores - demand_penalty
    
    return combined_heuristics