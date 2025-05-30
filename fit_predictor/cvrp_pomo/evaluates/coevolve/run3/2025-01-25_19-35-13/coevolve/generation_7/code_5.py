import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize demands relative to vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Calculate the sum of normalized demands for each edge
    edge_demand_sums = torch.dot(normalized_demands, distance_matrix)
    
    # Incorporate factors such as distance and road quality
    # Assuming road_quality_matrix is available, which is a tensor of the same shape as distance_matrix
    # road_quality_matrix = ... (to be provided)
    # road_quality_factor = road_quality_matrix / road_quality_matrix.sum()  # Normalize road quality
    # edge_quality_sums = torch.dot(road_quality_factor, distance_matrix)
    
    # Combine demand sums and distance to create the potential function
    # For simplicity, we will only use the demand sums in this example
    potential = edge_demand_sums
    
    # Introduce a robust potential function to prevent division by zero
    # This is a simple example using max to avoid negative values
    potential = torch.clamp(potential, min=0)
    
    return potential