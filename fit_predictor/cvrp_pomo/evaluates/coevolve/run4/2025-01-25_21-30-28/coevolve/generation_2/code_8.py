import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands from the end to the start
    cumulative_demands = torch.cumsum(demands[::-1], dim=0)[::-1]
    
    # Calculate the maximum load for each node (from the end to the start)
    max_loads = torch.clamp(cumulative_demands - demands, min=0)
    
    # Normalize the max_loads by the total vehicle capacity
    normalized_max_loads = max_loads / demands.sum()
    
    # Calculate the heuristic values based on the normalized max_loads
    # Negative heuristic values for undesirable edges
    heuristic_matrix = -normalized_max_loads
    
    # Positive heuristic values for promising edges
    # We can add a small positive value to avoid zero values
    positive_value = 1e-5
    heuristic_matrix = torch.where(heuristic_matrix <= 0, positive_value, heuristic_matrix)
    
    return heuristic_matrix