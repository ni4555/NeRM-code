import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check if the inputs are on the same device (e.g., CPU or GPU)
    if not (distance_matrix.is_cuda == demands.is_cuda):
        raise ValueError("The distance matrix and demands must be on the same device.")

    # Number of nodes
    n = distance_matrix.shape[0]

    # Initialize heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the demand per unit distance
    demand_per_unit_distance = demands / distance_matrix

    # Calculate the cost of not visiting each node, which is inversely proportional to the demand per unit distance
    cost_of_not_visiting = 1.0 / (demand_per_unit_distance + 1e-8)  # Add a small epsilon to avoid division by zero

    # Calculate the cost of visiting each node (1.0 because it costs 1 to visit a node)
    cost_of_visiting = torch.ones_like(demand_per_unit_distance)

    # Combine the costs into the heuristics matrix
    heuristics = cost_of_not_visiting - cost_of_visiting

    return heuristics