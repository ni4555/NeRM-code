import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()

    # Normalize the customer demands by the total vehicle capacity
    normalized_demands = demands / vehicle_capacity

    # Create a matrix to hold the heuristics
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Calculate the initial heuristic value for each edge
    # The heuristic is the negative of the normalized demand of the destination node
    heuristics_matrix[:, 1:] = -normalized_demands[1:]

    # Add the heuristic value for the edge from the depot to the first customer
    heuristics_matrix[0, 1] = -normalized_demands[1]

    return heuristics_matrix