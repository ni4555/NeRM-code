import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the sum of all demands to represent the fraction of vehicle capacity used
    demand_penalty_factor = 1 / demands.sum()
    normalized_demands = demands * demand_penalty_factor

    # Calculate the inverse distance heuristic (IDH)
    # The IDH is a simple heuristic where the weight of an edge is inversely proportional to the distance
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero

    # Calculate the demand penalty
    # The penalty is higher for edges that lead to vehicles close to their capacity
    demand_penalty = normalized_demands * distance_matrix

    # Combine the inverse distance and demand penalty to get the heuristic value for each edge
    heuristics = inverse_distance - demand_penalty

    return heuristics