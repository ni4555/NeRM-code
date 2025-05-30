import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    sum_of_demands = torch.sum(demands)
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / sum_of_demands
    
    # Compute the heuristic value for each edge
    # The heuristic is a function of the distance and the demand
    # For simplicity, here we use a simple heuristic: negative demand multiplied by distance
    # This is just an example, the actual heuristic would be more complex and tailored to the problem specifics
    heuristic_matrix = -normalized_demands.unsqueeze(1) * distance_matrix
    
    # The matrix should be of the same shape as the distance matrix
    assert heuristic_matrix.shape == distance_matrix.shape, "The heuristic matrix should have the same shape as the distance matrix."
    
    return heuristic_matrix