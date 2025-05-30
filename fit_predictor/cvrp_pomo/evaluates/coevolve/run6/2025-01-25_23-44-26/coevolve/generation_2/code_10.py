import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to the range [0, 1]
    demands_normalized = demands / demands.sum()
    
    # Create a distance matrix with the same shape as the input
    # where each cell represents the normalized distance from the depot to the customer
    distance_normalized = distance_matrix / distance_matrix.sum(axis=0, keepdim=True)
    
    # Compute the heuristic values by combining normalized distance and normalized demand
    # The heuristic value is the product of these two normalized values
    heuristics = distance_normalized * demands_normalized
    
    # The resulting heuristics matrix should have positive values for promising edges
    # and negative values for undesirable edges.
    # To achieve this, we can add a small constant to ensure no zero values and then
    # subtract the sum of the row to ensure all values are negative or positive
    epsilon = 1e-5
    heuristics = heuristics + epsilon - epsilon * (distance_normalized.sum(dim=1, keepdim=True) * demands_normalized)
    
    return heuristics