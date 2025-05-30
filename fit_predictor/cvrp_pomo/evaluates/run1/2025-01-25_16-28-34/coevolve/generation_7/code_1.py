import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Add a column for the depot demand
    demands = torch.cat([torch.zeros(1), demands], dim=0)

    # Initialize a matrix of the same shape as distance_matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # For edges from the depot to customers (i.e., edges to index 1 to n)
    heuristics[0, 1:] = 1 / distance_matrix[0, 1:] + demands[1:] / (demands.sum() * distance_matrix[0, 1:])

    # For edges between customers (i.e., edges to index 1 to n, excluding the depot)
    # We do not consider the depot to depot edge, so we start from index 1 and end at index n
    heuristics[1:, 1:] = 1 / distance_matrix[1:, 1:] + demands[1:] / (demands.sum() * distance_matrix[1:, 1:])

    # The matrix should be symmetric, so fill in the lower triangle
    heuristics = heuristics + heuristics.t() - torch.diag(torch.diag(heuristics))

    return heuristics

# Example usage:
distance_matrix = torch.tensor([[0, 2, 10, 6, 1],
                                [2, 0, 3, 8, 5],
                                [10, 3, 0, 7, 4],
                                [6, 8, 7, 0, 9],
                                [1, 5, 4, 9, 0]], dtype=torch.float32)
demands = torch.tensor([0, 1, 3, 2, 4], dtype=torch.float32)

heuristic_scores = heuristics_v2(distance_matrix, demands)
print(heuristic_scores)