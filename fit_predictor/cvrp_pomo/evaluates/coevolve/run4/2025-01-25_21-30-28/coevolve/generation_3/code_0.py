import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the initial heuristic values based on normalized demands
    heuristic_values = normalized_demands * (1 / distance_matrix)
    
    # Apply dynamic programming to adjust heuristic values based on edge weights
    for i in range(n):
        for j in range(n):
            if i != j:
                # If edge i-j is part of a shorter route, update the heuristic
                min_prev_heuristic = heuristic_values[i].min()
                heuristic_values[j] = min(min_prev_heuristic, heuristic_values[j])
    
    # Employ genetic algorithm-inspired technique to balance vehicle loads
    # Here, we use a simple approach of penalizing heavily the edges that would
    # cause an imbalance in the load
    load_balance_penalty = torch.abs(normalized_demands - demands / (n - 1))
    heuristic_values += load_balance_penalty
    
    return heuristic_values