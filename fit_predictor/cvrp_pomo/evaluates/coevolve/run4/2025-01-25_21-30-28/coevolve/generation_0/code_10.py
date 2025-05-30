import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands, subtract the demand of the depot, and normalize by vehicle capacity
    total_demand = demands.sum()
    normalized_demand = (demands - demands[0]).sum() / (total_demand - demands[0])

    # Create a matrix with all edges initialized to zero
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # For each customer, compute the potential heuristics based on distance and normalized demand
    for i in range(1, len(demands)):
        # Calculate the distance from the depot to each customer
        dist_to_customer = distance_matrix[0, i]
        # Compute the heuristics value
        heuristics = dist_to_customer - normalized_demand
        # Assign the heuristics value to the edge
        heuristics_matrix[0, i] = heuristics
        heuristics_matrix[i, 0] = heuristics  # The return edge to the depot also has the same heuristics value

    return heuristics_matrix