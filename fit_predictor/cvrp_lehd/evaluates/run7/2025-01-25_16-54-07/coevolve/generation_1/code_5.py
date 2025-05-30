import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the relative demand for each edge
    # For the diagonal elements (edges to the depot), use a large negative value to discourage them
    # For other edges, compute the negative product of the normalized demand at the source and destination
    relative_demand = -torch.where(distance_matrix != distance_matrix, torch.ones_like(distance_matrix), 
                                   torch.cat((torch.full_like(distance_matrix, -1e5), 
                                               torch.prod(normalized_demands[distance_matrix != distance_matrix], axis=0)), 
                                            dim=0))

    # For the diagonal elements (edges to the depot), set the value to a large positive value to encourage them
    # This will make the depot as a starting or ending point of a route more promising
    relative_demand[torch.arange(distance_matrix.shape[0]), torch.arange(distance_matrix.shape[0])] = 1e5

    return relative_demand