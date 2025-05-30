import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands to the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of the negative demands for each edge
    negative_demand_sum = -torch.sum(normalized_demands[:, None] * distance_matrix, dim=1)
    
    # Calculate the sum of the positive demands for each edge
    positive_demand_sum = torch.sum(normalized_demands[:, None] * distance_matrix, dim=1)
    
    # Combine the negative and positive demand sums to create the heuristic values
    heuristics = negative_demand_sum + positive_demand_sum
    
    return heuristics