import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    max_demand_ratio = torch.max(normalized_demands)
    
    # Create a tensor of the same shape as the distance matrix, initialized to -inf
    heuristic_values = torch.full(distance_matrix.shape, float('-inf'), dtype=torch.float)
    
    # Add a positive heuristic for the edges where demand is higher than the max demand ratio
    heuristic_values[distance_matrix != 0] = (normalized_demands[distance_matrix != 0] - max_demand_ratio) * (1 - demands[distance_matrix != 0])
    
    # Optionally, include more sophisticated heuristics like demand balancing or other rules
    # ...

    return heuristic_values