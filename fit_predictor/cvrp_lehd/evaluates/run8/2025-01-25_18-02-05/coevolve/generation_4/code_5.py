import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the sum of the product of distance and demand for each edge
    edge_potentials = (distance_matrix * normalized_demands).sum(dim=1)

    # Apply a simple heuristic: larger sum of distance-weighted demand is better
    # Here we use a negative sign to indicate that larger values are more promising
    heuristics = -edge_potentials

    return heuristics