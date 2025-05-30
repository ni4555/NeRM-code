import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the cost of not serving a customer, which is proportional to their demand
    not_serving_cost = -normalized_demands

    # Initialize the heuristics matrix with the not_serving_cost
    heuristics = torch.full_like(distance_matrix, fill_value=not_serving_cost)

    # Add a penalty for edges leading from the depot to non-customers
    for i in range(1, n):
        heuristics[0, i] = float('inf')

    # Add a penalty for edges leading from non-customers to the depot
    for i in range(1, n):
        heuristics[i, 0] = float('inf')

    # Add a penalty for edges between non-adjacent customers
    for i in range(n):
        for j in range(n):
            if i != j and i != 0 and j != 0 and distance_matrix[i, j] != float('inf'):
                heuristics[i, j] += (abs(i - j) * normalized_demands[i] + normalized_demands[j])

    return heuristics