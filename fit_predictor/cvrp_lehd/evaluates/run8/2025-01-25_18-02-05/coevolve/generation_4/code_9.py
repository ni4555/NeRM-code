import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for each customer
    normalized_demands = demands / demands.sum()
    
    # Compute the heuristic value for each edge
    # This is a simple example heuristic based on normalized demand, which is not necessarily optimal
    # but serves as an illustration of how to implement a vectorized heuristic function.
    heuristic_matrix = torch.outer(normalized_demands, normalized_demands)
    heuristic_matrix = heuristic_matrix * distance_matrix
    
    # Apply a simple threshold to encourage shorter paths by giving larger penalties to edges with higher distance
    threshold = 1.0
    heuristic_matrix[distance_matrix > threshold] = -torch.inf
    
    return heuristic_matrix