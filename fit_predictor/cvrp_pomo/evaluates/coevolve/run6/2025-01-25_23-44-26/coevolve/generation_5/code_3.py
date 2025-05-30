import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demand by the total vehicle capacity
    normalized_demands = demands / demands.sum()
    
    # Inverse distance heuristic
    inverse_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero
    
    # Demand penalty function
    demand_penalty = normalized_demands * (1 + demands / (n * 0.5))  # Scale demand by a factor
    
    # Combine the heuristics
    combined_heuristic = inverse_distance * demand_penalty
    
    # Normalize the combined heuristic to ensure non-negative values
    max_value = combined_heuristic.max()
    min_value = combined_heuristic.min()
    combined_heuristic = (combined_heuristic - min_value) / (max_value - min_value)
    
    return combined_heuristic