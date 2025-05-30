import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands_normalized = demands / demands.sum()
    
    # Calculate the cost of traveling from each node to every other node
    # Subtracting the normalized demand from the distance gives a heuristic value
    heuristics = distance_matrix - demands_normalized.unsqueeze(1) - demands_normalized.unsqueeze(0)
    
    # Apply swap-insertion heuristic
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Check if swap is valid and if the insertion would create a shorter path
                if (distance_matrix[i, k] + distance_matrix[k, j] < distance_matrix[i, j] and
                    (distance_matrix[i, k] - distance_matrix[i, j]) + (distance_matrix[k, j] - distance_matrix[i, k]) < 0):
                    heuristics[i, j] += 1
                    heuristics[j, i] -= 1

    # Apply 2-opt heuristic
    for k in range(n):
        for l in range(n):
            for m in range(n):
                for n in range(n):
                    # Check if 2-opt improvement is possible
                    if (distance_matrix[k, l] + distance_matrix[l, m] + distance_matrix[m, n] +
                        distance_matrix[n, k] < distance_matrix[k, l] + distance_matrix[l, n] +
                        distance_matrix[n, k] + distance_matrix[k, m] + distance_matrix[m, l]):
                        heuristics[k, l] -= 1
                        heuristics[l, k] += 1
                        heuristics[l, m] -= 1
                        heuristics[m, l] += 1
                        heuristics[m, n] -= 1
                        heuristics[n, m] += 1
                        heuristics[n, k] -= 1
                        heuristics[k, n] += 1
    
    return heuristics