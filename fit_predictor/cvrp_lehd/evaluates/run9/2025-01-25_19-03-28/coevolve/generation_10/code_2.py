import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands by total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse of demands to represent load balance
    inverse_demands = 1 / (normalized_demands + 1e-6)  # Add a small value to avoid division by zero
    
    # Initialize the heuristics matrix with high negative values for undesirable edges
    heuristics = -torch.ones_like(distance_matrix)
    
    # Calculate the distance-based heuristic
    heuristics = heuristics + distance_matrix
    
    # Incorporate demand-based heuristic to favor load balance
    heuristics = heuristics - normalized_demands
    
    # Incorporate inverse demand-based heuristic to favor load balance
    heuristics = heuristics + inverse_demands
    
    # Incorporate service time-based heuristic (assumed to be 1 for simplicity)
    heuristics = heuristics + 1
    
    # Ensure that the heuristics have a non-negative value
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics