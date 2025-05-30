import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demand between each pair of nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the absolute difference to handle both positive and negative demand differences
    abs_demand_diff = torch.abs(demand_diff)
    
    # Create a penalty matrix where edges with high demand differences are penalized
    penalty_matrix = abs_demand_diff * 1000  # This factor can be adjusted
    
    # Subtract the penalty from the distance matrix to get the heuristic values
    heuristics = distance_matrix - penalty_matrix
    
    # Set the diagonal to a large negative value to avoid selecting the depot as a customer
    torch.fill_diagonal_(heuristics, -float('inf'))
    
    return heuristics