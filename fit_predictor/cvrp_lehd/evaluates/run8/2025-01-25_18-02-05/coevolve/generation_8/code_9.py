import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands divided by the total capacity (for normalization)
    total_capacity = demands.sum()
    
    # Calculate the difference between each demand and the average demand
    avg_demand = demands / total_capacity
    demand_diff = demands - avg_demand
    
    # Calculate the inverse of the distance matrix (using torch.where to handle zeros)
    inv_distance = torch.where(distance_matrix != 0, 1.0 / distance_matrix, torch.tensor(float('inf')))
    
    # Combine the heuristics using the demand difference and inverse distance
    # The heuristic is a weighted sum of the inverse distance and the demand difference
    # Negative values are used for undesirable edges, positive values for promising ones
    heuristics = -inv_distance * demand_diff
    
    # Normalize the heuristics to ensure a consistent scale
    heuristics = heuristics / (heuristics.abs().max() + 1e-10)
    
    return heuristics