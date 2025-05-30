import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand along the diagonal, which represents the depot to customer distances
    cumulative_demand = torch.diag(demands)
    
    # Calculate the potential additional demand for each edge by adding the next customer's demand
    additional_demand = distance_matrix * demands
    
    # Normalize the additional demand to the vehicle capacity, which is implicitly considered as 1
    normalized_demand = additional_demand / (cumulative_demand + additional_demand)
    
    # Calculate the heuristic values: negative for undesirable edges, positive for promising ones
    heuristic_values = 1 - normalized_demand  # Promising edges should have higher weight, thus subtract from 1
    
    return heuristic_values