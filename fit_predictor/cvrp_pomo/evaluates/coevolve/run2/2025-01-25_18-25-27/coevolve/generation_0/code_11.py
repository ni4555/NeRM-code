import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()

    # Calculate the negative of distances for undesirable edges
    undesirable_edges = -distance_matrix

    # Calculate the positive heuristic values for promising edges
    # The heuristic is based on the fraction of remaining capacity when visiting a node
    # The higher the demand, the more negative the heuristic value (undesirable)
    # We normalize the demand by the total vehicle capacity
    remaining_capacity = 1 - (demands / total_demand)
    promising_edges = remaining_capacity * distance_matrix

    # Combine the undesirable and promising edges to form the final heuristics matrix
    heuristics_matrix = undesirable_edges + promising_edges

    # Ensure the heuristics values are within the required range (-1 to 1)
    heuristics_matrix = torch.clamp(heuristics_matrix, min=-1, max=1)

    return heuristics_matrix