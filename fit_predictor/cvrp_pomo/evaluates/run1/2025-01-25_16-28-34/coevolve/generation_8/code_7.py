import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the cost for each edge
    # Since the depot node is indexed by 0, the cost for the depot to itself is ignored
    # Cost is a sum of distance and normalized demand (since it's normalized, we just use it as is)
    cost_matrix = distance_matrix + demands
    
    # Apply a threshold to make edges negative for undesirable edges and positive for promising ones
    # This threshold can be adjusted based on the context
    threshold = torch.max(torch.abs(cost_matrix)) / 2
    heuristics_matrix = torch.where(cost_matrix > threshold, cost_matrix, -cost_matrix)
    
    return heuristics_matrix