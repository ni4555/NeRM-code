import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the maximum distance to ensure all distances are within [0, 1]
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance

    # Normalize the demands by the sum of demands to scale them between 0 and 1
    sum_of_demands = torch.sum(demands)
    normalized_demands = demands / sum_of_demands

    # Calculate the inverse distance heuristic (IDH) which is a measure of the attractiveness of visiting a node
    # Higher values indicate more attractive (shorter distance)
    inverse_distance_heuristic = 1 / normalized_distance_matrix

    # Integrate demand-penalty mechanism to deter overloading vehicles
    # Negative values are introduced to avoid nodes with high demands
    demand_penalty = -normalized_demands

    # Combine the IDH and demand penalty to get the initial heuristic values
    initial_heuristic = inverse_distance_heuristic + demand_penalty

    # Normalize the initial heuristic values to be between 0 and 1
    min_value = torch.min(initial_heuristic)
    max_value = torch.max(initial_heuristic)
    normalized_initial_heuristic = (initial_heuristic - min_value) / (max_value - min_value)

    return normalized_initial_heuristic