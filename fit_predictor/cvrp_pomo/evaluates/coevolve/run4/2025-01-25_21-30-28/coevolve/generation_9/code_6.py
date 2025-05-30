import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance heuristic
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Calculate the demand normalization heuristic
    total_demand = demands.sum()
    demand_normalized = demands / total_demand
    
    # Combine the heuristics
    combined_heuristics = inverse_distance * demand_normalized
    
    # Ensure that the heuristics are negative for undesirable edges and positive for promising ones
    # We can do this by subtracting the maximum value from the combined heuristics
    max_heuristic = combined_heuristics.max()
    heuristics = combined_heuristics - max_heuristic
    
    return heuristics