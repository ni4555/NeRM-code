import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    normalized_demands = demands / demands.sum()
    
    # Calculate the inverse distance heuristic ((IDH) values)
    idh_values = 1 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero
    
    # Incorporate demand-penalty mechanism
    demand_penalty = -10 * (normalized_demands[torch.arange(len(demands)), torch.arange(len(demands))] - 0.5)
    
    # Combine IDH and demand-penalty to get initial heuristics
    initial_heuristics = idh_values + demand_penalty
    
    # Normalize heuristics to have non-negative values
    initial_heuristics = (initial_heuristics - initial_heuristics.min()) / (initial_heuristics.max() - initial_heuristics.min())
    
    return initial_heuristics