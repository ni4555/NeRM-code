import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics
    # For the heuristic, we will use a simple demand-based heuristic where we
    # calculate the negative demand of the customer as it is more likely to be
    # included in the route if the demand is high. The idea is that edges leading
    # to customers with high demands will have a higher "cost" (lower heuristic value).
    heuristic_matrix = -normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)
    
    return heuristic_matrix