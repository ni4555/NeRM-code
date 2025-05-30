import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Calculate the potential value for each edge
    # Assuming demand relaxation and node partitioning
    # Potential value = distance_matrix * demand / total_capacity
    potential_value_matrix = distance_matrix * (demands / total_capacity)

    # Adjust the potential value for the depot (0,0)
    # By adding a large negative value to make it undesirable
    # This avoids starting the route from the depot if it's not needed
    depot_edge = torch.tensor([[0, 0], [0, 0]], dtype=distance_matrix.dtype)
    potential_value_matrix[depot_edge] -= float('inf')

    # Return the potential value matrix
    return potential_value_matrix