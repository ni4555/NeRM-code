import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with a large negative value for all edges
    heuristics = -torch.ones_like(distance_matrix)
    
    # Apply a normalization technique to level the demand of customer nodes
    # For simplicity, we'll use the demand as the heuristic value
    heuristics[distance_matrix != 0] = normalized_demands[distance_matrix != 0]
    
    # Apply a constraint-aware optimization process to manage vehicle capacities
    # For simplicity, we'll assume that the heuristics already reflect the capacity constraints
    # since the demands are normalized by the total capacity
    
    # No need for a dynamic adjustment of neighborhood structure as it's part of the hybrid evolutionary strategy
    # and not directly related to the heuristic function
    
    return heuristics