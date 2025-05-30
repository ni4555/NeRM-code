import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming that the distance_matrix is a torch.Tensor of shape (n, n)
    # and the demands is a torch.Tensor of shape (n,), where n is the number of nodes.
    
    # Initialize the heuristics matrix with zeros of the same shape as distance_matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Normalize demands by the total vehicle capacity for comparison
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of the squared differences between the distance and demand
    # We use the squared difference as a cost metric to penalize long distances and unbalanced loads
    squared_cost = (distance_matrix ** 2) + (distance_matrix * normalized_demands) ** 2
    
    # Use negative cost as an indicator for undesirable edges
    heuristics = -squared_cost
    
    return heuristics