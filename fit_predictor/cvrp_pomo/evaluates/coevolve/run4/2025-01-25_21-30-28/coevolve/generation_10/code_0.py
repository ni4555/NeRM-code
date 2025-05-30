import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of all demands to normalize them
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Calculate the inverse distance heuristic
    inv_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Calculate the Normalization heuristic (demand-based)
    normalization = normalized_demands * inv_distance
    
    # Combine the heuristics
    combined_heuristic = normalization
    
    # Ensure that the values are negative for undesirable edges and positive for promising ones
    combined_heuristic = combined_heuristic - combined_heuristic.max()
    
    return combined_heuristic