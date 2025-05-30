import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    # Normalize demands
    normalized_demands = demands / total_demand

    # Create a matrix of potential negative values for undesirable edges
    undesirable_edges = -torch.ones_like(distance_matrix)

    # Calculate the potential heuristics for each edge
    # For each edge, the heuristic is the negative of the demand multiplied by the distance
    heuristics = -normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)

    # Combine the negative potential with the matrix of potential negative values
    heuristics = torch.maximum(heuristics, undesirable_edges)

    return heuristics