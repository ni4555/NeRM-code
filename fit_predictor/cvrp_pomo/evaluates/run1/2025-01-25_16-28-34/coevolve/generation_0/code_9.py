import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand of all customers
    total_demand = demands.sum()
    
    # Calculate the normalized demand for each customer
    normalized_demands = demands / total_demand
    
    # Calculate the normalized distance for each edge
    normalized_distances = distance_matrix / distance_matrix.max()
    
    # Calculate the potential of each edge based on the normalized demand and distance
    potential = normalized_demands.unsqueeze(1) * normalized_distances.unsqueeze(0)
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-10
    potential = potential / (potential + epsilon)
    
    # Subtract the potential from 1 to get the heuristics (negative values for undesirable edges)
    heuristics = 1 - potential
    
    return heuristics