import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (sum of demands)
    total_capacity = demands.sum()
    
    # Create a tensor to store heuristics, initialized to 0
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the difference between total capacity and each customer's demand
    demand_diff = total_capacity - demands
    
    # For each edge, calculate the potential heuristic value
    # Promising edges have positive values, undesirable edges have negative values
    # We use the maximum of 0 to avoid negative values which indicate undesirable edges
    heuristics = torch.max(demand_diff, torch.zeros_like(demand_diff))
    
    return heuristics