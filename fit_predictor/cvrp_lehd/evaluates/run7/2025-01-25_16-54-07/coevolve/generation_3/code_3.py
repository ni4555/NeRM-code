import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands have the same shape
    assert distance_matrix.shape == demands.shape, "Distance matrix and demands must have the same shape."
    
    # Calculate the normalized demand for each customer
    normalized_demands = demands / demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the savings for each edge
    savings_matrix = (distance_matrix[:, 1:] + distance_matrix[1:, :] - 2 * distance_matrix) * normalized_demands[1:]
    
    # Subtract the service time from the savings
    savings_matrix -= 1
    
    # Add the savings to the diagonal (edges to itself)
    torch.fill_diagonal(savings_matrix, 0)
    
    # Use the savings matrix to calculate the heuristics
    heuristics_matrix[1:, 1:] = savings_matrix
    
    return heuristics_matrix