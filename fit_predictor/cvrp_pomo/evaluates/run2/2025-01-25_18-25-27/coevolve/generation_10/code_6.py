import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values based on normalized demands
    heuristic_matrix = distance_matrix * normalized_demands
    
    # Apply a normalization technique to level the demand of customer nodes
    # We can use min-max normalization for this purpose
    min_demand = normalized_demands.min()
    max_demand = normalized_demands.max()
    normalized_demand = (normalized_demands - min_demand) / (max_demand - min_demand)
    
    # Adjust the heuristic values to be within the range of [0, 1]
    heuristic_matrix = (heuristic_matrix - min_demand) / (max_demand - min_demand)
    
    return heuristic_matrix