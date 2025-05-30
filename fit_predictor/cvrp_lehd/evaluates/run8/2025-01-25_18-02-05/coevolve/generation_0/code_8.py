import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the sum of all demands
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum

    # Create a vector that represents the sum of the product of distances and demands for each edge
    edge_potentials = distance_matrix * normalized_demands

    # Compute the maximum potential for each edge (maximize the heuristics)
    max_potentials = edge_potentials.max(dim=1)[0]

    # Add a small positive constant to avoid division by zero
    positive_constant = 1e-8
    max_potentials = max_potentials + positive_constant

    # Calculate the inverse of the potentials to get the heuristics
    heuristics = 1.0 / max_potentials

    return heuristics