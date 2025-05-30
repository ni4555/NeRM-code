import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Create a matrix where each element is the sum of the normalized demand and normalized distance
    # This will be used to calculate the heuristics
    demand_distance_sum = normalized_demands.unsqueeze(1) + normalized_distance_matrix.unsqueeze(0)
    
    # The heuristics are calculated by subtracting the sum from 1 to get negative values for undesirable edges
    heuristics = 1 - demand_distance_sum
    
    return heuristics