import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the negative of the distance matrix as a heuristic
    negative_distance_matrix = -distance_matrix
    
    # Use the normalized demands to scale the distance matrix
    # This heuristic function will favor shorter distances to nodes with higher demands
    scaled_distance_matrix = negative_distance_matrix * normalized_demands
    
    return scaled_distance_matrix