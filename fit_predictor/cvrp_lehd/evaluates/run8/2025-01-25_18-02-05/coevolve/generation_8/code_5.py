import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensors are on the same device and have the same dtype
    distance_matrix = distance_matrix.to(demands.device).to(demands.dtype)
    demands = demands.to(distance_matrix.device).to(distance_matrix.dtype)
    
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity (in this case, we normalize by the total demand)
    normalized_demands = demands / total_demand
    
    # Create a tensor of the same size as the distance matrix with all elements initialized to 0
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values based on the demands and distances
    # For simplicity, we use a linear combination of the demand and the inverse of the distance
    # This is just an example, the actual heuristic could be more complex
    heuristics = normalized_demands * distance_matrix / (distance_matrix + 1e-8)
    
    # Set the diagonal elements to a very low value to avoid choosing the depot as an edge
    torch.fill_diagonal_(heuristics, float('-inf'))
    
    return heuristics