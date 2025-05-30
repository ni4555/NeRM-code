import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Inverse Distance Heuristic (IDH) - assign customers based on the reciprocal of their distance
    # We subtract the distances to ensure higher weights for closer customers
    idh_weights = 1 / (distance_matrix[:, 1:] - distance_matrix[:, 0][:, None])
    idh_weights[distance_matrix[:, 1:] - distance_matrix[:, 0][:, None] == 0] = 0  # Avoid division by zero
    
    # Demand penalty function - increase cost for assigning high-demand customers to vehicles near capacity
    demand_penalty = normalized_demands[1:] * demands[1:]
    
    # Combine IDH and demand penalty into a single heuristic matrix
    # We add the penalties to the weights since we want to penalize higher costs
    heuristic_matrix = idh_weights + demand_penalty
    
    return heuristic_matrix