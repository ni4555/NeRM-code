import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate normalized distance matrix by dividing each distance by the max distance
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the weighted demand for each edge based on demand and normalized distance
    # The weights are calculated as demand divided by the sum of all demands, which represents the fraction of capacity for each customer
    weights = demands / demands.sum()
    weighted_demand_matrix = weights.view(1, -1) * weights.view(-1, 1) * normalized_distance_matrix
    
    # Subtract the weighted demand matrix from 1 to get the negative heuristics
    heuristics = 1 - weighted_demand_matrix
    
    return heuristics