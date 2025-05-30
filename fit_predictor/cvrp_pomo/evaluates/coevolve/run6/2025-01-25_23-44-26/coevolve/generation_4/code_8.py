import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity (assuming demands are already normalized by capacity)
    total_capacity = demands.sum()
    
    # Inverse distance heuristic: calculate the reciprocal of the distance
    inv_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Demand penalty function: increase cost for edges leading to vehicles close to capacity
    # We calculate the penalty based on the reciprocal of the demand to favor lower demand customers
    demand_penalty = 1.0 / (demands + 1e-8)
    capacity_penalty = (demands / total_capacity) * (1 - 1 / (demands + 1e-8))
    penalty = capacity_penalty * (inv_distance + 1e-8)
    
    # Combine heuristics: inverse distance and demand penalty
    combined_heuristics = inv_distance - penalty
    
    # Ensure negative values for undesirable edges and positive values for promising ones
    combined_heuristics = combined_heuristics.clamp(min=-1e8, max=1e8)
    
    return combined_heuristics