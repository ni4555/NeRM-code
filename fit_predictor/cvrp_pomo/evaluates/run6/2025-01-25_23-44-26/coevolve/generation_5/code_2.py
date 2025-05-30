import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demand by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Demand penalty function: higher penalty for higher normalized demand
    demand_penalty = normalized_demands * 10  # Example factor, adjust as needed
    
    # Inverse distance heuristic: lower distance, higher heuristic value
    inverse_distance = 1 / (distance_matrix ** 2)  # Squared to ensure non-zero values
    
    # Combine both heuristics
    combined_heuristic = inverse_distance - demand_penalty
    
    # Avoid negative values by clamping
    combined_heuristic = torch.clamp(combined_heuristic, min=0)
    
    return combined_heuristic