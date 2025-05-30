import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (assuming all vehicles have the same capacity)
    total_capacity = demands.sum()
    
    # Normalize customer demands to the range [0, 1]
    normalized_demands = demands / total_capacity
    
    # Initialize a torch tensor with the same shape as distance_matrix, filled with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Example heuristic: Promote edges with lower distance and lower demand difference
    # The following is a placeholder for the actual heuristic logic
    heuristics += -distance_matrix  # Lower distances are better, thus negative value
    heuristics += (1 - torch.abs(normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)))  # Promote edges with similar demands
    
    # This is a simple example heuristic, in a real-world scenario you would replace this
    # with a more complex heuristic that takes into account all the requirements mentioned.
    
    return heuristics