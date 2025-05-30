import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand mask
    cumulative_demand_mask = demands.cumsum(dim=0)
    
    # Calculate the normalized demand mask
    normalized_demand_mask = cumulative_demand_mask / cumulative_demand_mask[-1]
    
    # Calculate the load distribution heuristic
    load_distribution_heuristic = normalized_demand_mask * (1 - demands)
    
    # Calculate the distance-based heuristic
    distance_heuristic = -distance_matrix
    
    # Combine the two heuristics
    combined_heuristic = load_distribution_heuristic + distance_heuristic
    
    return combined_heuristic