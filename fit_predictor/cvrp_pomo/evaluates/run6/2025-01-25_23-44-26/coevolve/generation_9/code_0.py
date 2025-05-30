import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    total_capacity = demands.sum()
    demands = demands / total_capacity
    
    # Inverse Distance Heuristic (IDH)
    # Calculate the inverse distance for each edge
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Demand-sensitive penalty mechanism
    # Calculate the demand-based penalty for each edge
    demand_penalty = demands.unsqueeze(1) * demands.unsqueeze(0)
    
    # Combine IDH and demand penalty to get the heuristic values
    heuristics = inverse_distance - demand_penalty
    
    return heuristics