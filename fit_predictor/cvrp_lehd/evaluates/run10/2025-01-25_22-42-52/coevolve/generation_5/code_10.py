import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensors are on the same device and are of the correct shape
    distance_matrix = distance_matrix.to(demands.device)
    demands = demands.to(demands.device)

    # The number of nodes is the size of the matrix
    n = distance_matrix.shape[0]

    # Create a tensor with the negative of the demands to use in the heuristic
    negative_demands = -demands

    # Compute the heuristic value for each edge (i, j)
    # The heuristic is a function of the distance and the difference in demands
    # Here, we use a simple heuristic that penalizes longer distances and higher demand differences
    heuristics = distance_matrix + negative_demands.unsqueeze(1) + negative_demands.unsqueeze(0)

    # We want to discourage longer distances and higher demand differences, so we subtract the sum of the demands
    # to normalize the heuristic values
    total_demand = torch.sum(demands)
    heuristics = heuristics - 2 * total_demand

    return heuristics