import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_normalized = demands / total_capacity
    
    # Inverse distance heuristic
    inv_distance = 1 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero
    
    # Demand normalization heuristic
    demand_normalized = demands / demands.sum()
    
    # Combine heuristics
    combined_heuristic = inv_distance * demand_normalized
    
    # Negative values for undesirable edges (to be minimized during the search)
    combined_heuristic = -combined_heuristic
    
    return combined_heuristic