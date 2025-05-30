import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Inverse Distance Heuristic (IDH)
    inverse_distance = 1 / (distance_matrix + 1e-10)  # Adding a small value to avoid division by zero
    
    # Demand-penalty mechanism
    demand_penalty = -demands
    
    # Combine IDH and demand-penalty
    combined_heuristic = inverse_distance + demand_penalty
    
    # Normalize the combined heuristic to ensure non-negative values
    min_val = combined_heuristic.min()
    max_val = combined_heuristic.max()
    normalized_heuristic = (combined_heuristic - min_val) / (max_val - min_val)
    
    return normalized_heuristic