import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Compute the negative of the distance matrix to encourage shorter paths
    negative_distance_matrix = -distance_matrix

    # Calculate the sum of demands for each destination node
    sum_of_demands = demands.sum(dim=0, keepdim=True)

    # Calculate the potential negative impact of exceeding vehicle capacity
    # We use the maximum possible load that a vehicle can carry (equal to the sum of demands)
    # multiplied by the negative distance, to penalize longer routes
    potential_excess_load_penalty = (demands * negative_distance_matrix).sum(dim=1, keepdim=True)

    # Normalize by the sum of demands to make the heuristic relative to the total capacity
    normalized_potential_excess_load_penalty = potential_excess_load_penalty / sum_of_demands

    # The heuristic function is a combination of the negative distance and the normalized penalty
    # We subtract the penalty from the negative distance to make higher values better
    heuristic_matrix = negative_distance_matrix - normalized_potential_excess_load_penalty

    return heuristic_matrix