import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are tensors and demand vector is on the same device as distance matrix
    distance_matrix = distance_matrix.to(torch.float32)
    demands = demands.to(torch.float32)

    # Calculate the total vehicle capacity by summing the normalized demands
    vehicle_capacity = demands.sum()

    # Initialize the heuristic matrix with zeros, same shape as the distance matrix
    heuristics = torch.zeros_like(distance_matrix)

    # For each customer node i, calculate the contribution to the heuristic for each edge (i, j)
    for i in range(len(demands)):
        # Calculate the demand contribution for each edge (i, j)
        for j in range(len(demands)):
            if i != j:
                demand_contribution = -demands[j]
                # Calculate the heuristic value for the edge (i, j)
                heuristics[i, j] = demand_contribution - distance_matrix[i, j]

    return heuristics