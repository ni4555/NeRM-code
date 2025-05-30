import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands divided by the total capacity to normalize
    total_capacity = demands.sum()
    # Normalize the demands by the total capacity
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    # Here we use a simple heuristic that considers the negative of the distance
    # multiplied by the ratio of the customer demand to the total capacity
    heuristics_matrix = -distance_matrix * normalized_demands
    
    # Clip the values to ensure that they are within a certain range
    # For example, we can ensure that the maximum value is 0 and the minimum is -1
    heuristics_matrix = torch.clamp(heuristics_matrix, min=-1.0, max=0.0)
    
    return heuristics_matrix