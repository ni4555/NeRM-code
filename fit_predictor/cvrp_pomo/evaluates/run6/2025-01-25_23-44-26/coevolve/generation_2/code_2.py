import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of normalized demands for each row (from the depot to each customer)
    row_sums = normalized_demands.sum(dim=1)
    
    # Calculate the sum of normalized demands for each column (from each customer to the depot)
    col_sums = normalized_demands.sum(dim=0)
    
    # Calculate the heuristic values for each edge
    # We want to discourage edges with high normalized demand, so we use negative values
    # We also want to discourage long distances, so we subtract the distance from the heuristic
    heuristics = row_sums - col_sums - distance_matrix
    
    return heuristics