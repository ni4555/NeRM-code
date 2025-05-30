import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize the matrix
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Calculate the difference between demands of each pair of customers
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the heuristic as the product of the normalized demand difference and the distance
    # We use a negative sign to encourage the algorithm to include edges with lower demand differences
    heuristics = -demand_diff * distance_matrix
    
    # Normalize the heuristics to have a similar scale as the original matrix
    heuristics /= heuristics.max()
    
    return heuristics