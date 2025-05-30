import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of all demands to normalize them
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics based on normalized demands
    # Here we use a simple heuristic that considers demand density
    # The idea is to promote edges with higher demand density
    heuristics = distance_matrix * normalized_demands
    
    return heuristics