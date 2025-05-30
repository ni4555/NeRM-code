import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristic matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the heuristic values
    # For simplicity, we use the normalized demand as the heuristic value
    heuristics = normalized_demands

    # Apply swap-insertion heuristic
    for i in range(1, n):
        for j in range(i + 1, n):
            # Compute the swap heuristic
            swap_heuristic = heuristics[i] + heuristics[j] - heuristics[i] - heuristics[j]
            # Update the heuristic matrix for the swapped edges
            heuristics[i, j] = swap_heuristic
            heuristics[j, i] = swap_heuristic

    # Apply 2-opt heuristic
    for k in range(2, n):
        for i in range(1, n - k + 1):
            for j in range(i + k, n + 1):
                # Compute the 2-opt heuristic
                two_opt_heuristic = heuristics[i, j] - heuristics[i, j - 1] - heuristics[i + 1, j] + heuristics[i, j - 1] + heuristics[i + 1, j - 1]
                # Update the heuristic matrix for the 2-opt edges
                heuristics[i, j] = two_opt_heuristic
                heuristics[i + 1, j] = two_opt_heuristic
                heuristics[j, i] = two_opt_heuristic
                heuristics[j, i + 1] = two_opt_heuristic

    return heuristics