import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for each node
    sum_demands = demands.sum(dim=1, keepdim=True)
    
    # Calculate the total capacity of all vehicles
    total_capacity = demands.sum()
    
    # Calculate the remaining capacity for each node after visiting all other nodes
    remaining_capacity = (sum_demands - demands) / (total_capacity - demands)
    
    # Calculate the potential benefit of visiting each node (promising edges)
    potential_benefit = remaining_capacity * demands
    
    # Calculate the cost of visiting each node (undesirable edges)
    cost = distance_matrix.sum(dim=1, keepdim=True) - distance_matrix
    
    # Combine the potential benefit and cost to get the heuristic values
    heuristics = potential_benefit - cost
    
    return heuristics