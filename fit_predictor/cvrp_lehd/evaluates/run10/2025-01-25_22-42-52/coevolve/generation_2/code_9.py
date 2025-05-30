import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demand vector by the total demand
    normalized_demands = demands / total_demand
    
    # Calculate the negative exponential of the normalized demands
    # This will give us the heuristic values for each edge
    heuristics = torch.exp(-normalized_demands)
    
    # To ensure the heuristic matrix has negative values for undesirable edges
    # We add a small constant to the exponentiated values
    small_constant = 1e-5
    heuristics = heuristics + small_constant
    
    # Replace any zero values with a very small negative value
    # to avoid numerical issues with zero heuristic values
    heuristics[distance_matrix == 0] = -small_constant
    
    return heuristics