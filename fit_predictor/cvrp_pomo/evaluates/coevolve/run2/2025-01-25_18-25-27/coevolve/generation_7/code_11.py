import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance matrix and demands are both of shape (n, n) and (n,) respectively
    assert distance_matrix.shape == (len(demands), len(demands))
    assert demands.shape == (len(demands),)

    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Calculate the cost of serving each customer, which is the distance to the customer
    # minus the normalized demand of that customer. This encourages selecting customers
    # that are close and have lower demand.
    cost_matrix = distance_matrix - normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)

    # The resulting cost matrix will have negative values for promising edges (since
    # distances are subtracted from demands), and zero or positive values for undesirable edges.
    return cost_matrix