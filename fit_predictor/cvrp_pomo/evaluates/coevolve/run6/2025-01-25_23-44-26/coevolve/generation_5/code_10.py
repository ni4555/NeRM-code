import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance heuristic
    inverse_distance = 1 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Demand penalty function: higher penalty for high-demand customers near vehicle capacity
    demand_penalty = demands * (1 - (1 / (1 + normalized_demands)))
    
    # Combine the heuristics
    combined_heuristics = inverse_distance - demand_penalty
    
    return combined_heuristics