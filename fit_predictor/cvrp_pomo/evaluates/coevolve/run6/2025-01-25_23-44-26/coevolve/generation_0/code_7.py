import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of demands along the diagonal (for self-loops)
    diagonal_sum = normalized_demands.sum(dim=0, keepdim=True)
    
    # Create a matrix where each cell is the difference between the sum of demands and the diagonal
    demand_diff_matrix = normalized_demands - diagonal_sum
    
    # Compute the sum of each row and add to the matrix (heuristic value)
    heuristic_values = demand_diff_matrix.sum(dim=1, keepdim=True)
    
    # Use a threshold to determine if an edge is promising or not
    # For simplicity, we can use a threshold of 0.5 for this example
    threshold = torch.tensor(0.5)
    
    # Initialize the result matrix with negative values (undesirable edges)
    heuristic_matrix = -torch.ones_like(distance_matrix)
    
    # Set the positive heuristic values
    heuristic_matrix[demand_diff_matrix > threshold] = heuristic_values[demand_diff_matrix > threshold]
    
    return heuristic_matrix