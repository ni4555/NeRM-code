import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the distance matrix as a heuristic
    negative_distance_matrix = -distance_matrix
    
    # Calculate the difference between demands to get the urgency of each customer
    demand_diff_matrix = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the absolute values of the differences to prioritize closer customers
    demand_diff_matrix = torch.abs(demand_diff_matrix)
    
    # Sum the urgency and the distance to get the total heuristic value for each edge
    heuristic_matrix = negative_distance_matrix + demand_diff_matrix
    
    return heuristic_matrix