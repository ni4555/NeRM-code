import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands divided by the total capacity
    demand_ratio = demands / demands.sum()
    
    # Calculate the sum of distances divided by the total capacity
    distance_ratio = distance_matrix / distance_matrix.sum()
    
    # Calculate the heuristic as the product of demand and distance ratios
    heuristic_matrix = demand_ratio * distance_ratio
    
    # Negative values indicate undesirable edges, positive values indicate promising ones
    # We can adjust the scale of heuristics to ensure negative values are present
    # by subtracting the maximum value from the entire matrix
    max_heuristic = heuristic_matrix.max()
    heuristic_matrix = heuristic_matrix - max_heuristic
    
    return heuristic_matrix