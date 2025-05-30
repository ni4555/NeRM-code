import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize a tensor of the same shape as the distance matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic value for each edge
    # Here we use a simple heuristic: the more demand, the more promising the edge
    # This is just a placeholder; the actual heuristic should be more complex and incorporate
    # dynamic window approaches, multi-objective evolutionary algorithms, etc.
    heuristic_matrix = -normalized_demands * distance_matrix
    
    # We could apply a smoothing function or a non-linear transformation to the heuristic matrix
    # to avoid overly large values and to encourage more balanced routes
    
    return heuristic_matrix