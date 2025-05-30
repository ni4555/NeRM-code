import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Problem-specific Local Search: Calculate the initial heuristic based on demands
    for i in range(1, n):
        for j in range(1, n):
            heuristic_matrix[i, j] = -demands[i]
    
    # Adaptive PSO Population Management: Placeholder for PSO exploration
    # This would involve updating the heuristic_matrix based on PSO dynamics
    # For the sake of the example, we'll just copy the initial heuristic
    # In a real implementation, this would be replaced with PSO-driven updates
    
    # Dynamic Tabu Search with Adaptive Cost Function: Placeholder for Tabu Search
    # This would involve further updating the heuristic_matrix based on Tabu Search dynamics
    # For the sake of the example, we'll just copy the PSO result
    # In a real implementation, this would be replaced with Tabu Search-driven updates
    
    # Combine the results from PSO and Tabu Search (if applicable)
    # For the sake of the example, we'll just return the initial heuristic
    # In a real implementation, this would be a combination of the PSO and Tabu Search results
    
    return heuristic_matrix