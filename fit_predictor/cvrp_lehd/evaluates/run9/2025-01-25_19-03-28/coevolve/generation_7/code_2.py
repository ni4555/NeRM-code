import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity (assuming all demands are given as fractions of capacity)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the potential of each edge to be included in the solution
    # We will use a simple heuristic that combines the normalized demand and distance
    # This is a simple approach that assumes higher demand and lower distance are more promising
    # You can replace this with a more complex heuristic as needed
    
    # Calculate the negative of the distance matrix for a more intuitive heuristic
    negative_distances = -distance_matrix
    
    # Combine the negative distances with the normalized demands
    combined_potential = negative_distances + normalized_demands
    
    return combined_potential