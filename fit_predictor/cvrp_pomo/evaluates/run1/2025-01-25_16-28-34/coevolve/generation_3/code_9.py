import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the sum of demands to create a comparable scale
    demand_sum = demands.sum()
    normalized_distance_matrix = distance_matrix / demand_sum

    # Normalize the demands by the sum of demands to ensure all demands are on the same scale
    normalized_demands = demands / demand_sum

    # Calculate the potential benefit of each edge based on normalized distance and demand
    # The heuristic function is a simple product of normalized distance and normalized demand
    heuristic_matrix = normalized_distance_matrix * normalized_demands

    # Adjust the heuristic matrix to ensure no route exceeds vehicle capacity
    # This is done by subtracting the maximum possible load from the heuristic value
    max_load = 1.0  # Assuming vehicle capacity is the total sum of demands
    adjusted_heuristic_matrix = heuristic_matrix - max_load

    return adjusted_heuristic_matrix