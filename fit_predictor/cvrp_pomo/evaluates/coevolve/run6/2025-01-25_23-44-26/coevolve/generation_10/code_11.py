import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Apply Inverse Distance Heuristic (IDH) for initial customer assignment
    # IDH assigns a negative value for each edge, which we invert to get positive values
    heuristics = -distance_matrix
    
    # Add demand-driven factor to the heuristics
    # This increases the weight of edges that are closer to the vehicle's capacity
    capacity_factor = 1 / (1 + demands)
    heuristics += capacity_factor
    
    # Normalize the heuristics matrix to ensure uniform problem scale
    heuristics /= heuristics.max()
    
    return heuristics