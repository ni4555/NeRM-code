import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized by dividing by the sum of demands
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum

    # Create a new matrix filled with ones for all possible distances
    heuristics = torch.ones_like(distance_matrix)

    # Calculate the heuristics matrix by taking the minimum distance and subtracting the normalized demand
    heuristics = heuristics * (distance_matrix - normalized_demands.unsqueeze(1))

    # Set diagonal values to a very negative value to avoid choosing the depot as an edge
    torch.fill_diagonal_(heuristics, -torch.inf)

    return heuristics