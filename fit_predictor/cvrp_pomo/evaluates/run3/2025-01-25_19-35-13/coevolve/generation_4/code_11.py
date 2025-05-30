import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand-to-distance ratio for each edge
    demand_to_distance = demands / distance_matrix
    
    # Normalize the ratio by subtracting the mean ratio to ensure a balanced scale
    mean_ratio = torch.mean(demand_to_distance)
    normalized_ratio = demand_to_distance - mean_ratio
    
    # Calculate the sum of the demand along each route
    demand_sum = torch.sum(demands, dim=0)
    
    # Calculate the heuristics values for each edge
    # Higher values indicate more promising edges
    heuristics = -normalized_ratio * (1 + demand_sum)
    
    return heuristics