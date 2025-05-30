import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity (assuming it's a single row vector)
    demand_sum = torch.sum(demands)
    normalized_demands = demands / demand_sum
    
    # Initialize a torch.Tensor to hold heuristics with the same shape as distance_matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Incorporate demand-based heuristic
    heuristics += normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Incorporate distance-based heuristic
    heuristics -= distance_matrix
    
    # Incorporate service time heuristic (example: assume 1 unit of service time per unit of demand)
    service_time = normalized_demands
    heuristics -= service_time.unsqueeze(1) * service_time.unsqueeze(0)
    
    # Normalize heuristics to have non-negative values
    heuristics = heuristics.clamp(min=0)
    
    return heuristics