import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Calculate the potential of each edge
    # We use a simple heuristic that considers the normalized demand and distance
    # Promising edges will have a higher potential (positive values)
    # Undesirable edges will have a lower potential (negative values)
    # Here we use a simple formula: potential = demand * distance
    # Adjusting the formula to ensure that the values are negative for undesirable edges
    potential = normalized_demands * distance_matrix
    
    # We want to encourage edges with lower distances and higher demands
    # To do this, we can add a large negative value to the edges with higher distances
    # This is a simple way to penalize longer distances
    # We use a large constant to ensure that the values are negative
    large_constant = 1e5
    distance_penalty = large_constant * (distance_matrix - distance_matrix.min(dim=1, keepdim=True)[0])
    potential += distance_penalty
    
    # The potential matrix will have negative values for undesirable edges
    # and positive values for promising ones
    return potential