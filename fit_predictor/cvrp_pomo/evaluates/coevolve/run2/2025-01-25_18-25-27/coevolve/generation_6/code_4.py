import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Normalize the distance matrix to make it comparable to the demands
    normalized_distance_matrix = distance_matrix / distance_matrix.max()

    # Calculate the heuristic value for each edge
    heuristics = normalized_distance_matrix - normalized_demands

    # Apply a penalty for edges that are longer than the average distance
    average_distance = normalized_distance_matrix.mean()
    heuristics[distance_matrix > average_distance] -= 1.0

    # Apply a penalty for edges that are carrying more than the average demand
    average_demand = normalized_demands.mean()
    heuristics[distance_matrix > average_demand] -= 1.0

    return heuristics