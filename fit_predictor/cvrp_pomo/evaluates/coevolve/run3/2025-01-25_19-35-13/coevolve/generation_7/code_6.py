import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity by summing the demands
    vehicle_capacity = demands.sum()
    
    # Normalize demands by vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Initialize potential function with zeros
    potential_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the sum of normalized demands for each edge
    sum_normalized_demands = torch.sum(normalized_demands[:, None] + normalized_demands[None, :], dim=2)
    
    # Incorporate distance and road quality factors
    # Assuming road_quality_matrix is a matrix with values representing road quality between nodes
    # road_quality_matrix = torch.Tensor([...]) # This should be defined outside this function
    # For demonstration, we'll use a constant value for road quality
    road_quality_factor = 1.0
    distance_factor = 1.0
    
    # Calculate the potential for each edge
    potential_matrix = (sum_normalized_demands * road_quality_factor) * distance_factor
    
    # Handle division by zero errors by adding a small epsilon
    epsilon = 1e-8
    potential_matrix = torch.clamp(potential_matrix, min=-epsilon, max=epsilon)
    
    return potential_matrix