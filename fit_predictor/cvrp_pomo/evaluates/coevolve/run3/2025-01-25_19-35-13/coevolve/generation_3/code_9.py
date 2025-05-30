import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / torch.sum(distance_matrix, dim=1, keepdim=True)
    # Calculate the potential function based on demand
    demand_potential = -demands
    # Combine normalized distance and demand potential to form the heuristic matrix
    heuristic_matrix = normalized_distance_matrix + demand_potential
    # Use an epsilon value to prevent division by zero
    epsilon = 1e-8
    # Ensure no division by zero occurs
    heuristic_matrix = torch.clamp(heuristic_matrix, min=-epsilon, max=epsilon)
    return heuristic_matrix