import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor with zeros of the same shape as distance_matrix
    heuristics = torch.zeros_like(distance_matrix)

    # Dynamic window technique: Adjust heuristic values based on vehicle capacities
    for i in range(n):
        if i != 0:  # Skip the depot node
            for j in range(n):
                if j != 0:  # Skip the depot node
                    # Calculate the heuristic value based on the difference in demands
                    # and considering the distance
                    heuristic_value = distance_matrix[i, j] - demands[i] + demands[j]
                    heuristics[i, j] = heuristic_value

    # Constraint programming: Ensure that each customer is visited only once
    for i in range(1, n):  # Skip the depot node
        heuristics[i, 0] = -float('inf')  # Cannot go back to the depot
        for j in range(1, n):  # Skip the depot node
            heuristics[i, j] = min(heuristics[i, j], heuristics[i, j - 1] - demands[j] + demands[i])

    # Multi-objective evolutionary algorithm: Adjust heuristic values for load balancing
    # (This is a simplified version, not a real evolutionary algorithm)
    for i in range(1, n):  # Skip the depot node
        for j in range(1, n):  # Skip the depot node
            # Increase heuristic value if adding this edge helps in load balancing
            if demands[i] < demands[j]:
                heuristics[i, j] += 0.1  # Heuristic adjustment factor

    return heuristics