import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with high values (undesirable edges)
    heuristic_matrix = torch.full((n, n), fill_value=1e5)
    
    # Greedy assignment based on demand
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                heuristic_matrix[i, j] = -normalized_demands[i] * distance_matrix[i, j]
    
    # Add depot to each customer's cost
    for i in range(1, n):
        heuristic_matrix[0, i] = -normalized_demands[i] * distance_matrix[0, i]
    
    # Local search to refine the solution
    for _ in range(10):  # Number of iterations for local search
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    for k in range(1, n):
                        if k != i and k != j:
                            # Calculate the potential change in the heuristic values
                            change_i_j = heuristic_matrix[i, j] - heuristic_matrix[i, k]
                            change_j_k = heuristic_matrix[j, k] - heuristic_matrix[i, k]
                            if change_i_j > 0 and change_j_k > 0:
                                # Update the heuristic matrix if the change is positive
                                heuristic_matrix[i, j] = change_i_j
                                heuristic_matrix[j, k] = change_j_k
    
    return heuristic_matrix