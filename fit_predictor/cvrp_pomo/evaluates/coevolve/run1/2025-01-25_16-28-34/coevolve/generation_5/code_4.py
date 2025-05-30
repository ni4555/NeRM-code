import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are of the same shape
    assert distance_matrix.shape == (len(demands), len(demands)), "Distance matrix and demands must be of the same shape."

    # The depot node is indexed by 0, so we need to subtract the distances to the depot from the distance matrix
    distance_to_depot = distance_matrix[:, 0].unsqueeze(1)
    distance_matrix_subtracted = distance_matrix - distance_to_depot

    # The heuristic is a weighted sum of the inverse distance and demand
    # We use a small epsilon to avoid division by zero
    epsilon = 1e-6
    heuristic_matrix = -torch.div(distance_matrix_subtracted, epsilon + distance_matrix_subtracted)

    # Normalize the heuristic by the sum of the demands to ensure a balanced score
    demand_sum = demands.sum()
    if demand_sum == 0:
        demand_sum = 1  # Avoid division by zero
    normalized_heuristic = heuristic_matrix * (demands / demand_sum)

    return normalized_heuristic