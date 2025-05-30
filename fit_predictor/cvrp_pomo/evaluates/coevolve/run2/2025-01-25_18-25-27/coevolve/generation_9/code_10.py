import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands[1:].sum()
    demands_normalized = demands[1:] / total_capacity
    
    # Calculate the heuristic for each edge
    # The heuristic will be a combination of distance and demand
    # Negative values for undesirable edges, positive for promising ones
    # We use a simple heuristic here: negative distance + demand
    # This heuristic assumes that the closer the customer, the more promising the edge is
    # and the lower the demand, the more promising it is.
    heuristics = -distance_matrix + demands_normalized
    
    # Ensure that the depot (node 0) has a very high heuristic value to avoid being visited
    heuristics[0] = float('inf')
    
    return heuristics