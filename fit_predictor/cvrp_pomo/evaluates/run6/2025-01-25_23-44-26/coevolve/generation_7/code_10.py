import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity
    
    # Inverse Distance heuristic
    inverse_distance = 1.0 / (distance_matrix + 1e-10)  # Adding a small value to avoid division by zero
    
    # Demand-penalty mechanism
    demand_penalty = demands_normalized - demands_normalized.min()
    
    # Combine heuristics
    combined_heuristic = inverse_distance * demand_penalty
    
    return combined_heuristic