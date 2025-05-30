import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Initialize a tensor with zeros to store heuristics values
    heuristics = torch.zeros_like(distance_matrix)
    
    # Normalize demand to the range [0, 1]
    normalized_demands = demands / demands.max()
    
    # Inverse distance heuristic
    heuristics = 1.0 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Demand penalty function: scale the cost for high-demand customers
    demand_penalty = (1 + demands * 0.1)  # Example penalty factor of 0.1 per unit of demand
    
    # Combine the heuristics with the demand penalty
    heuristics *= demand_penalty
    
    # Normalize the heuristics values to ensure non-negative values
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics