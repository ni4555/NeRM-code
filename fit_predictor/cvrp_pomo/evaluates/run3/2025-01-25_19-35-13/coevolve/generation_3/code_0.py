import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are of the same size
    assert distance_matrix.shape[0] == distance_matrix.shape[1] == demands.shape[0], "Mismatched dimensions"
    
    # Normalize the demands by the total vehicle capacity (assuming total capacity is 1 for simplicity)
    normalized_demands = demands / demands.sum()
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the potential function based on distance and demand
    # The potential function could be a weighted sum of the inverse of distance and the normalized demand
    # Here, we use a simple example where we weigh distance inversely with demand
    potential_function = 1 / (distance_matrix + 1e-10)  # Adding epsilon to avoid division by zero
    potential_function *= normalized_demands.unsqueeze(1)  # Unsqueeze for broadcasting
    
    # Compute the heuristic values by taking the negative of the potential function
    # Negative values are undesirable edges, positive values are promising ones
    heuristic_matrix = -potential_function
    
    return heuristic_matrix