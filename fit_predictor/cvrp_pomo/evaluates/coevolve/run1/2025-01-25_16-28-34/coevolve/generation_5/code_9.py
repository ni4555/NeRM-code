import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the total demand for all customers
    total_demand = demands[1:].sum()  # Exclude the depot demand
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values based on the normalized demands
    # The heuristic is a combination of the normalized demand and the distance
    # We use a simple heuristic where we subtract the normalized demand from the distance
    # to favor edges with lower demands
    heuristics = distance_matrix - (normalized_demands[1:] * distance_matrix[1:, 1:])
    
    # The depot node should have a high heuristic value to avoid being visited
    heuristics[0, 1:] = -1e6
    heuristics[1:, 0] = -1e6
    
    return heuristics