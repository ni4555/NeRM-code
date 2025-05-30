import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized distance matrix by dividing each distance by the maximum distance
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance

    # Calculate the load on the vehicle for each customer
    total_demand = torch.sum(demands)
    load_matrix = normalized_distance_matrix * demands

    # Calculate the total load on each route if the current edge is included
    # The sum is taken along the columns (customers) and then divided by the total demand to normalize
    total_load_matrix = torch.sum(load_matrix, dim=1) / total_demand

    # Calculate the cost of including each edge based on the load and distance
    # We want to minimize the cost, so we take the difference between 1 and the load and multiply by the distance
    edge_cost_matrix = (1 - total_load_matrix) * normalized_distance_matrix

    # Calculate the heuristics values
    # We want to promote edges with lower cost, so we take the negative of the cost
    heuristics_matrix = -torch.sum(edge_cost_matrix, dim=1)

    return heuristics_matrix