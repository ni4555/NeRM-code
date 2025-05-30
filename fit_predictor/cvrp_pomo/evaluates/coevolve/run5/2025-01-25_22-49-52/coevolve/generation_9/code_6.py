import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the potential of each edge
    # Here we use a simple heuristic based on normalized demand and distance
    # This is just an example and can be replaced with more sophisticated methods
    potential = -distance_matrix * normalized_demands
    
    # Add a small constant to avoid division by zero when taking the log
    potential = potential + 1e-10
    
    # Apply the normalized demand to the potential
    potential = potential / potential.sum() * demands.sum()
    
    return potential