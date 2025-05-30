import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance matrix is of shape (n, n) and the demands vector is of shape (n,)
    n = distance_matrix.shape[0]
    assert distance_matrix.shape == (n, n), "Distance matrix must be of shape (n, n)."
    assert demands.shape == (n,), "Demands vector must be of shape (n,)."

    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()

    # Normalize demands to vehicle capacity
    normalized_demands = demands / vehicle_capacity

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the potential cost of serving each customer
    # Negative values indicate undesirable edges (customers with high demands)
    potential_costs = -normalized_demands

    # Apply a simple heuristic: the lower the distance, the more promising the edge
    # Here we assume that the shortest distance to a customer is more promising
    heuristics += distance_matrix

    # Adjust the heuristics for demand
    heuristics += potential_costs

    # Normalize the heuristics to ensure the sum of values for each row is the same
    # This step is to prevent any single customer from dominating the solution
    row_sums = heuristics.sum(dim=1, keepdim=True)
    heuristics = heuristics / row_sums

    return heuristics