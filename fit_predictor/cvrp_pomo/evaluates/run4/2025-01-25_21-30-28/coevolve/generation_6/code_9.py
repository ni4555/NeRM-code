import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse of the distance matrix
    inv_distance_matrix = torch.reciprocal(distance_matrix)
    
    # Normalize the demands to sum to the total vehicle capacity (which is 1 in this case)
    normalized_demands = demands / demands.sum()
    
    # Multiply the inverse distances by the normalized demands
    # This will give us higher values for edges closer to the demands
    heuristics = inv_distance_matrix * normalized_demands
    
    # Subtract the demand from each element to make it negative where the demand is high
    # We want to avoid visiting nodes with high demands unless necessary
    heuristics -= demands
    
    # Return the resulting heuristics matrix
    return heuristics