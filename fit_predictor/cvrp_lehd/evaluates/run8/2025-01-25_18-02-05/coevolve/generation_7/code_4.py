import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (sum of demands)
    total_capacity = demands.sum()
    
    # Normalize the demands to represent the fraction of vehicle capacity required by each customer
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values as the negative of the distance (undesirable edges)
    # and add a small positive value for the edges that are not from the depot to avoid zero values
    heuristics = -distance_matrix + 1e-5 * (distance_matrix != 0)
    
    # Adjust the heuristics based on the normalized demands
    heuristics += normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)
    
    return heuristics