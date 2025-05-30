import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    epsilon = 1e-10  # To prevent division by zero
    total_capacity = demands.sum()
    demand_normalized = demands / total_capacity
    
    # Calculate the potential function based on distance and demand
    potential = -distance_matrix + demand_normalized.unsqueeze(1) * distance_matrix
    
    # Normalize the potential to prevent overflow and to guide the search towards promising edges
    max_potential = torch.max(potential, dim=1)[0]
    normalized_potential = (potential - max_potential.unsqueeze(1)) / (max_potential + epsilon)
    
    # Introduce a bias for the depot to encourage starting from it
    depot_index = 0
    normalized_potential[depot_index, :] += 1
    
    return normalized_potential