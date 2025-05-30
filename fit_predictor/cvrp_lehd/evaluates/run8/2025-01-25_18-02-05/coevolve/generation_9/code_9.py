import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to represent the fraction of the total capacity each customer requires
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost of serving each customer
    potential_costs = distance_matrix * normalized_demands
    
    # Subtract the potential costs from 1 to get the heuristic values
    # Positive values indicate promising edges, negative values undesirable edges
    heuristics = 1 - potential_costs
    
    return heuristics