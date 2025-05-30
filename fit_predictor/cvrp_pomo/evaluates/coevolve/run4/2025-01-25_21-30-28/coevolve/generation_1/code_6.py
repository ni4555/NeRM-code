import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost of visiting each customer
    # The heuristic is a combination of the distance and the normalized demand
    # The idea is to penalize longer distances and high demands
    potential_costs = distance_matrix * normalized_demands
    
    # The heuristics function returns negative values for undesirable edges
    # and positive values for promising ones.
    # We can simply return the potential_costs matrix which is already in the desired format.
    return potential_costs