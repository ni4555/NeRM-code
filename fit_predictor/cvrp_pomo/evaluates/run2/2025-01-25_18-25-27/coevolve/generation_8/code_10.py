import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the initial heuristics based on normalized demands
    heuristics += demands_normalized

    # Apply swap-insertion heuristic
    for i in range(1, n):
        for j in range(i + 1, n):
            # Calculate potential improvement if we swap the edges (i, k) and (j, k)
            # where k is the common node between (i, k) and (j, k)
            for k in range(1, n):
                if distance_matrix[i, k] != 0 and distance_matrix[j, k] != 0:
                    improvement_i = -distance_matrix[i, k] * demands_normalized[k]
                    improvement_j = -distance_matrix[j, k] * demands_normalized[k]
                    improvement_swap = (distance_matrix[i, j] - distance_matrix[j, i]) * demands_normalized[k]
                    potential_improvement = improvement_i + improvement_j + improvement_swap

                    # Update the heuristics if the swap is beneficial
                    heuristics[i, j] += potential_improvement
                    heuristics[j, i] += potential_improvement

    # Apply 2-opt heuristic
    for k in range(1, n):
        for i in range(1, n):
            for j in range(i + 1, n):
                # Calculate the improvement if we reverse the sub-route between i and j
                improvement = 0
                for l in range(i, j):
                    improvement -= distance_matrix[l, l + 1]
                    improvement += distance_matrix[l + 1, l]
                improvement *= demands_normalized[k]

                # Update the heuristics if the reversal is beneficial
                heuristics[i, j] += improvement
                heuristics[j, i] += improvement

    # Incorporate real-time penalties to prevent overloading
    for i in range(1, n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] != 0:
                # Calculate the penalty based on the potential load of the route
                potential_load = demands_normalized[i] + demands_normalized[j]
                if potential_load > 1:
                    penalty = (potential_load - 1) * 10  # Example penalty factor
                    heuristics[i, j] -= penalty
                    heuristics[j, i] -= penalty

    return heuristics