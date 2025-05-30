import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the "promise" for each edge, which is the negative of the distance
    # multiplied by the demand (since we want negative values for undesirable edges)
    edge_promise = -distance_matrix * normalized_demands

    # We can enhance the heuristic by considering the capacity constraints.
    # For example, we could add a term that encourages visiting customers with higher demands.
    # However, since the problem statement mentions a dynamic capacity allocation, we will
    # simply return the negative distance as the heuristic value.

    return edge_promise