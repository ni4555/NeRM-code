import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Inverse distance heuristic: Prioritize edges with lower distance
    inverse_distance = 1 / (distance_matrix + 1e-10)  # Add small constant to avoid division by zero
    
    # Demand normalization heuristic: Adjust the priority based on demand
    demand_adjustment = normalized_demands.unsqueeze(0).unsqueeze(1) * normalized_demands.unsqueeze(1).unsqueeze(0)
    
    # Combine heuristics
    combined_heuristics = inverse_distance - demand_adjustment
    
    return combined_heuristics