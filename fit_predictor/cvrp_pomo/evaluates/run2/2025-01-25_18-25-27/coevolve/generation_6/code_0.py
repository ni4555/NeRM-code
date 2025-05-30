import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity
    
    # Initialize heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate distance-based heuristics
    heuristics += distance_matrix
    
    # Adjust for customer demand
    heuristics += demands_normalized
    
    # Apply normalization technique to equalize customer demands
    heuristics /= heuristics.max()
    
    # Apply constraint-aware process to maximize efficiency
    # Assuming that shorter distances are better and demand is a penalty
    heuristics = -heuristics
    
    return heuristics