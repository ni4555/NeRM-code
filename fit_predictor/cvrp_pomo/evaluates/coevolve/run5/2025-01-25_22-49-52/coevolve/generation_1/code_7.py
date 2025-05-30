import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands = demands.to(torch.float32) / demands.sum()  # Normalize demands
    
    # Calculate the negative of the distance matrix for the heuristic
    neg_distance_matrix = -distance_matrix
    
    # Incorporate demand relaxation into the heuristic
    demand_relaxation = demands.unsqueeze(1) * demands.unsqueeze(0)
    
    # Combine the two elements into the heuristic matrix
    heuristic_matrix = neg_distance_matrix + demand_relaxation
    
    # Add a small constant to ensure no division by zero
    epsilon = 1e-8
    heuristic_matrix = (heuristic_matrix - epsilon).div(epsilon)
    
    # Return the heuristic matrix, with high values for undesirable edges and low values for promising ones
    return heuristic_matrix