import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands to have a sum of 1
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic values using the Euclidean TSP heuristic
    # h(i, j) = a * distance(i, j) + b * demand(i) * demand(j)
    # where a and b are tunable parameters
    a = 1.0
    b = 0.5
    
    # Compute the heuristics matrix
    heuristics = a * distance_matrix + b * (normalized_demands[:, None] * normalized_demands[None, :])
    
    # Add a penalty to the diagonal elements to avoid visiting the depot twice
    heuristics = heuristics - distance_matrix.diagonal()
    
    return heuristics