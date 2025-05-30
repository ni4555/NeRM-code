import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance_matrix and demands are on the same device
    if distance_matrix.device != demands.device:
        demands = demands.to(distance_matrix.device)
    
    # Calculate the sum of demands to normalize the cost
    total_demand = demands.sum()
    
    # Normalize demands to represent the load per unit distance
    normalized_demands = demands / total_demand
    
    # Calculate the initial heuristic values based on normalized demands
    # Here we are using a simple heuristic where the load on each edge is used
    # as a heuristic value. Lower values indicate more promising edges.
    heuristic_matrix = -normalized_demands * distance_matrix
    
    return heuristic_matrix