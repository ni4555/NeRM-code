import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the demands for the depot node are set to 0, so we can ignore them in the computation.
    # Compute the sum of the demands from the depot to all other nodes
    sum_demands = demands[1:]  # Skip the demand for the depot (index 0)

    # Compute the negative of the distance to use it for heuristic value calculation
    negative_distances = -distance_matrix

    # Use a simple heuristic where the more distant edges get higher negative heuristic values
    # (meaning they are less desirable, which translates to a higher positive value for the heuristic)
    heuristic_values = negative_distances + sum_demands

    return heuristic_values

# Example usage:
# Assuming distance_matrix and demands are defined as follows:
# distance_matrix = torch.tensor([[0, 2, 5, 10],
#                                  [2, 0, 3, 7],
#                                  [5, 3, 0, 2],
#                                  [10, 7, 2, 0]], dtype=torch.float32)
# demands = torch.tensor([0, 3, 2, 4], dtype=torch.float32)
# heuristics_matrix = heuristics_v2(distance_matrix, demands)
# print(heuristics_matrix)