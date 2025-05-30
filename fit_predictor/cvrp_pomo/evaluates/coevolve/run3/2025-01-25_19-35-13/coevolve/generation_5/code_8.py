import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the initial heuristic values as the negative of the normalized demands
    # to prioritize edges with lower demands.
    initial_heuristics = -normalized_demands

    # Apply a simple heuristic to improve the initial heuristics:
    # If the demand is zero, we should not consider this edge as it's not needed.
    zero_demand_mask = (normalized_demands == 0).float()
    initial_heuristics += zero_demand_mask * 1e6  # penalize zero demand edges

    # Another heuristic could be to penalize edges that would exceed the vehicle's capacity
    # when added to the current route. For simplicity, we use the sum of demands to estimate
    # the remaining capacity, which is not the most accurate approach but serves as an example.
    remaining_capacity = total_capacity
    for i in range(n):
        for j in range(n):
            if i != j:
                # If adding this edge exceeds the remaining capacity, penalize it
                if remaining_capacity < demands[j]:
                    initial_heuristics[i, j] += 1e6

                # Update the remaining capacity after considering this edge
                remaining_capacity -= demands[j]

    return initial_heuristics