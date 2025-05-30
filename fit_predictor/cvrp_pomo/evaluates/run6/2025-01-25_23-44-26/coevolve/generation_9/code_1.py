import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse of the distance matrix
    inv_distance_matrix = 1 / (distance_matrix + 1e-8)  # Add a small value to avoid division by zero
    
    # Normalize the demands by the sum of all demands to get the capacity load per customer
    demand_ratio = demands / demands.sum()
    
    # Compute the heuristics by combining inverse distance and demand-sensitive ratio
    heuristics = -inv_distance_matrix * demand_ratio
    
    # Clip negative values to a very small number to represent undesirable edges
    heuristics = torch.clamp(heuristics, min=-1e-8)
    
    return heuristics