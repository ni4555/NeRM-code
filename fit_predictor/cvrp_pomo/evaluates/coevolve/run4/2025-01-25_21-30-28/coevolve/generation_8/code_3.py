import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Inverse distance heuristic
    inv_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Demand normalization heuristic
    demand_normalized = demands / demands.sum()
    
    # Combining heuristics
    combined_heuristics = inv_distance * demand_normalized
    
    # Ensuring that the values are within a reasonable range to prevent overflow or underflow
    combined_heuristics = torch.clamp(combined_heuristics, min=-1.0, max=1.0)
    
    return combined_heuristics