import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity (assuming demands are already normalized)
    total_capacity = demands.sum()
    
    # Calculate the sum of normalized demands for each edge
    edge_demand_sum = torch.dot(demands, distance_matrix.t())
    
    # Incorporate distance and road quality factors
    # Assuming road_quality_matrix is a precomputed matrix with road quality values
    # road_quality_matrix = torch.tensor([...])  # Replace with actual road quality matrix
    # edge_quality_factor = torch.dot(road_quality_matrix, distance_matrix.t())
    # edge_demand_sum = edge_demand_sum + edge_quality_factor
    
    # Calculate the potential function
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    potential = edge_demand_sum / (distance_matrix + epsilon)
    
    # Apply a small penalty to the diagonal elements (no movement from the depot)
    penalty = torch.eye(distance_matrix.shape[0]) * -1e8
    potential = potential + penalty
    
    return potential