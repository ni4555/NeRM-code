import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Inverse Distance Heuristic (IDH) - the heuristic function will be inversely proportional to the distance
    # and will also consider the demand. The demand-sensitive penalty mechanism will be incorporated here.
    
    # Calculate the inverse of the distance matrix (edges with higher distance will have lower weights)
    inv_distance_matrix = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Normalize the demand vector by the total vehicle capacity (assuming the capacity is 1 for simplicity)
    normalized_demands = demands / demands.sum()
    
    # Create a matrix that combines the inverse distance and the demand normalization
    # The demand-sensitive penalty will be a linear combination of the normalized demand and the inverse distance
    # We use a penalty factor to control how much we penalize high demands
    penalty_factor = 1.0
    heuristic_matrix = inv_distance_matrix * (1 + penalty_factor * normalized_demands)
    
    return heuristic_matrix