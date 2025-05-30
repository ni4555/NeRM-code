import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (sum of customer demands)
    total_capacity = demands.sum().item()
    
    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the "promise" of including each edge in a solution
    # A simple heuristic could be the product of the distance and the normalized demand
    promise_matrix = distance_matrix * normalized_demands.unsqueeze(1)
    
    # We want negative values for undesirable edges and positive values for promising ones.
    # To ensure this, we add a large positive number to the undesirable edges.
    # In this example, we consider edges that are not from the depot to be undesirable.
    undesirable_mask = (distance_matrix != 0) & (distance_matrix != distance_matrix[:, 0][:, None])
    promise_matrix[undesirable_mask] += torch.finfo(torch.float32).max
    
    return promise_matrix