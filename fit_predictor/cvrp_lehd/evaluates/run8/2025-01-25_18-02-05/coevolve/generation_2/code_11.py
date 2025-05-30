import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized
    demands = demands / demands.sum()

    # Initialize a tensor with the same shape as the distance matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the cost of each edge (distance * demand)
    cost_matrix = distance_matrix * demands.unsqueeze(1) * demands.unsqueeze(0)

    # Calculate the minimum cost for each edge to include it in the solution
    min_cost_matrix = cost_matrix.min(dim=1)[0].unsqueeze(1)
    min_cost_matrix = torch.cat((min_cost_matrix, min_cost_matrix), dim=1)
    min_cost_matrix = torch.cat((min_cost_matrix, min_cost_matrix.unsqueeze(0)), dim=0)

    # Subtract the minimum cost from the original cost matrix to get the heuristics
    heuristics = cost_matrix - min_cost_matrix

    return heuristics