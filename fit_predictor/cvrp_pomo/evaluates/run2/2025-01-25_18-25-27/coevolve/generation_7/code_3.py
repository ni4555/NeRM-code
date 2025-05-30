import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the total vehicle capacity is the sum of demands (excluding the depot demand)
    total_capacity = torch.sum(demands)
    
    # Normalize customer demands to have a sum of total_capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the demand contribution to the heuristic value
    demand_contrib = (normalized_demands[1:] - normalized_demands[:-1]) * distance_matrix[1:, :-1]
    
    # Calculate the capacity constraint contribution
    capacity_contrib = torch.clamp(distance_matrix[1:, :-1] / total_capacity, min=0)
    
    # Combine both contributions
    combined_contrib = demand_contrib - capacity_contrib
    
    return combined_contrib