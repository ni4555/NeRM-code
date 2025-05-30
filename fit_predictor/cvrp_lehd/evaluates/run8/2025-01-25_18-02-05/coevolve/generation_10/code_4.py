import random
import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of customer demands
    vehicle_capacity = demands.sum()

    # Compute the normalized demands, which are customer demands divided by the total vehicle capacity
    normalized_demands = demands / vehicle_capacity

    # Generate a random matrix with the same shape as distance_matrix
    random_matrix = torch.rand_like(distance_matrix)

    # Calculate the potential for each edge by subtracting the random component from the distance
    potential = distance_matrix - random_matrix

    # Adjust the potential by adding the normalized demands
    adjusted_potential = potential + normalized_demands.unsqueeze(1)

    # Generate a heuristics matrix that contains negative values for undesirable edges
    # and positive values for promising ones by subtracting a small constant to make
    # larger distances less likely to be chosen (promising edges)
    heuristics = adjusted_potential - torch.min(adjusted_potential, dim=1, keepdim=True)[0]

    return heuristics