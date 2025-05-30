import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensors are on the same device
    distance_matrix = distance_matrix.to(demands.device)
    demands = demands.to(demands.device)
    
    # Normalize demands to be between 0 and 1
    max_demand = demands.max()
    normalized_demands = demands / max_demand
    
    # Compute path potential based on distance and demand
    path_potential = distance_matrix * normalized_demands
    
    # Add a penalty for high demand to encourage load balancing
    demand_penalty = 1 - normalized_demands
    penalized_potential = path_potential * demand_penalty
    
    # Normalize the potential to ensure values are within a consistent scale
    max_potential = penalized_potential.max()
    normalized_potential = penalized_potential / max_potential
    
    # Apply a heuristic to adjust the potential for promising edges
    # Here, we assume a simple heuristic that promotes edges with lower potential
    heuristic_factor = torch.exp(-normalized_potential)
    
    # Return the adjusted potential as the heuristic value
    return heuristic_factor