import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity
    
    # Calculate the potential function as a weighted sum of distance and demand
    potential = distance_matrix * demands_normalized
    
    # Normalize the potential function to ensure no division by zero
    epsilon = 1e-10
    potential = torch.clamp(potential, min=-epsilon, max=epsilon)
    
    # The heuristics value for each edge is the negative of the potential function
    heuristics = -potential
    
    return heuristics