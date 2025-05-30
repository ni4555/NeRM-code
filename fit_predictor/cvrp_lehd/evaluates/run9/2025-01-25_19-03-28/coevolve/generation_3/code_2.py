import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are both of the same shape
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
    assert distance_matrix.shape[0] == demands.shape[0]
    
    # Calculate the maximum distance in the matrix to use as a threshold
    max_distance = distance_matrix.max()
    
    # Normalize the distance matrix by dividing by the maximum distance
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Normalize the demands by dividing by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristics by subtracting the normalized demands from the normalized distances
    heuristics = normalized_distance_matrix - normalized_demands
    
    # For the diagonal elements (distance from the depot to itself), we set them to a large negative value
    # to make them less likely to be chosen (the depot is not a customer)
    diagonal_mask = torch.eye(distance_matrix.shape[0], dtype=torch.bool)
    heuristics[diagonal_mask] = -float('inf')
    
    return heuristics