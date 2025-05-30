import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance_matrix and demands are tensors
    distance_matrix = distance_matrix.clone()
    demands = demands.clone()
    
    # Normalize the distance matrix by the total demand to account for the vehicle capacity
    # This assumes that the demands have already been normalized by the total vehicle capacity
    # and that the distance matrix does not include the distance from the depot to itself.
    distance_matrix[distance_matrix == 0] = torch.max(distance_matrix)
    distance_matrix /= demands
    
    # Add a large negative value for the diagonal to avoid selecting the depot to itself
    distance_matrix.fill_diagonal_(-1e10)
    
    # The heuristic score will be the negative distance to promote selecting this edge
    return -distance_matrix

# Example usage:
# distance_matrix = torch.tensor([[0, 1.5, 3.0, 2.5], [1.5, 0, 4.0, 3.0], [3.0, 4.0, 0, 1.0], [2.5, 3.0, 1.0, 0]])
# demands = torch.tensor([0.5, 0.6, 0.3, 0.2])  # Normalized demands
# heuristics = heuristics_v2(distance_matrix, demands)
# print(heuristics)