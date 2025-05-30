import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Inverse Distance Heuristic (IDH)
    inv_distance_heuristic = 1.0 / (distance_matrix + 1e-6)  # Add small constant to avoid division by zero
    
    # Demand-Penalty Heuristic
    demand_penalty = -demands
    
    # Normalize the demand penalty to the same scale as the inverse distance heuristic
    max_demand_penalty = torch.max(demand_penalty)
    normalized_demand_penalty = demand_penalty / max_demand_penalty
    
    # Combine the heuristics
    combined_heuristic = inv_distance_heuristic + normalized_demand_penalty
    
    # Apply a scaling factor to ensure that the heuristics are within a manageable range
    scaling_factor = 1.0 / (torch.max(combined_heuristic) + 1e-6)
    scaled_combined_heuristic = combined_heuristic * scaling_factor
    
    return scaled_combined_heuristic