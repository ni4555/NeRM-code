import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential value heuristic for each edge
    # Here, we are using a simple approach that assumes that the potential value is inversely proportional
    # to the demand of the destination node (since higher demand is harder to fulfill).
    # This is just a placeholder for a more sophisticated potential calculation.
    potential_values = 1 / (1 + normalized_demands)  # This could be replaced with a more complex formula

    # Calculate the potential difference between all pairs of nodes
    # This will be used to compute the edge heuristics
    potential_differences = potential_values.unsqueeze(0) - potential_values.unsqueeze(1)

    # Compute the negative distance matrix for the heuristic calculation
    # (We use negative values because we want to minimize the sum of heuristics)
    neg_distance_matrix = -distance_matrix

    # Calculate the edge heuristics as the sum of the potential difference and the negative distance
    heuristics = neg_distance_matrix + potential_differences

    # Fill in the diagonal with a very large negative value to ensure that no edge to the same node is considered
    heuristics.fill_diagonal_(torch.tensor(-float('inf')))

    return heuristics