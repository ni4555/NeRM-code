import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demand vector to have a sum of 1
    normalized_demands = demands / demands.sum()
    
    # Calculate the cumulative demand matrix
    cumulative_demand_matrix = torch.cumsum(normalized_demands.unsqueeze(1) * distance_matrix, dim=0)
    
    # Calculate the negative cumulative demand to discourage longer routes
    negative_cumulative_demand = -cumulative_demand_matrix
    
    # Normalize the matrix to ensure the values are in a scale suitable for the heuristic
    # Assuming the sum of demands is equal to the capacity of a vehicle for normalization
    max_demand = negative_cumulative_demand.max()
    heuristics_matrix = negative_cumulative_demand / max_demand
    
    return heuristics_matrix