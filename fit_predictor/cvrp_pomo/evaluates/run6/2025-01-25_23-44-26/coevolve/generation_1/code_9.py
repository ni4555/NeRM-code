import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix
    distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the sum of demands for each customer
    demand_sums = demands.cumsum()
    
    # Calculate the difference between the sum of demands and the total capacity
    demand_diffs = demand_sums - demands
    
    # Calculate the heuristics value for each edge
    heuristics = distance_matrix - demand_diffs
    
    return heuristics