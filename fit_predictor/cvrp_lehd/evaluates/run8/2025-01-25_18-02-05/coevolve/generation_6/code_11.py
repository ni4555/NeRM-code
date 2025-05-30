import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized (sum of demands equals 1)
    if not torch.isclose(torch.sum(demands), torch.tensor(1.0)):
        raise ValueError("Demands must sum up to 1 (normalized).")
    
    # Ensure that the distance_matrix and demands are valid for the problem
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be a square matrix.")
    if demands.ndim != 1 or demands.shape[0] != distance_matrix.shape[0]:
        raise ValueError("Demands must be a 1D vector with the same length as the distance matrix.")
    
    # Calculate the heuristic values using the savings algorithm
    n = distance_matrix.shape[0]
    savings_matrix = torch.full((n, n), float('-inf'))
    
    # The savings for connecting each customer to the next one, excluding the depot
    for i in range(1, n):
        for j in range(i+1, n):
            savings_matrix[i, j] = demands[i] + demands[j] - (distance_matrix[0, i] + distance_matrix[i, j] + distance_matrix[j, 0])
            savings_matrix[j, i] = savings_matrix[i, j]  # Since the matrix is symmetric
    
    # Replace negative savings with zero to mark them as undesirable
    savings_matrix = torch.clamp(savings_matrix, min=0)
    
    return savings_matrix