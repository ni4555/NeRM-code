import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity

    # Initialize heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the heuristic for each edge based on distance and demand
    # Here we use a simple heuristic: the sum of the distance and the normalized demand
    heuristics = distance_matrix + demands_normalized

    # Apply penalties for edges leading to overloading
    # This is a simple example; more complex logic can be used
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the total demand on the route if this edge is included
                total_demand = demands_normalized[i] + demands_normalized[j]
                # Apply a penalty if the total demand exceeds the capacity
                if total_demand > 1.0:
                    heuristics[i, j] -= 1000  # Arbitrary large penalty

    return heuristics