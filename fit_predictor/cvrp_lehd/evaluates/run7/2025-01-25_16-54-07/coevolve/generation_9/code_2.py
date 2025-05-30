import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand along each edge
    cumulative_demand = (demands[:, None] + demands[None, :]) / 2
    
    # Calculate the heuristics as the negative of the cumulative demand
    heuristics = -cumulative_demand
    
    # Normalize the heuristics by the maximum possible demand on a vehicle
    max_demand = torch.max(demands)
    heuristics /= max_demand
    
    # Subtract the distance matrix to get negative values for undesirable edges
    heuristics -= distance_matrix
    
    return heuristics