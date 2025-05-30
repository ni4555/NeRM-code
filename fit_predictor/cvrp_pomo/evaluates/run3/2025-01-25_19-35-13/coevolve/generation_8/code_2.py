import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / vehicle_capacity
    
    # Compute the sum of normalized demands for each edge
    edge_demand_sums = (normalized_demands[:-1] + normalized_demands[1:]) * distance_matrix
    
    # Compute the negative sum of distances to encourage shorter routes
    negative_distance_sums = -edge_demand_sums.sum(dim=0)
    
    # Add a term to encourage respecting vehicle capacities
    capacity_term = -torch.clamp(torch.abs(demands[1:]), min=0, max=vehicle_capacity)
    
    # Combine terms and add small constant to avoid division by zero
    heuristic_values = negative_distance_sums + capacity_term + 1e-8
    
    return heuristic_values