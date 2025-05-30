import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the absolute sum of the distance matrix to find the maximum distance
    max_distance = torch.max(torch.sum(distance_matrix, dim=0), dim=1).values

    # Normalize demands with respect to vehicle capacity
    demands = demands / torch.max(demands)

    # Initialize a matrix with the same shape as distance_matrix with all elements set to zero
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Loop through each customer (not including the depot) to calculate heuristic for each edge
    for i in range(1, distance_matrix.shape[0]):
        # Calculate the heuristic as a function of demand and distance
        heuristics_matrix[:, i] = demands[i] - distance_matrix[:, i]

    # For each customer, adjust the heuristic based on the maximum distance
    heuristics_matrix += max_distance * demands

    # Return the matrix containing heuristics values
    return heuristics_matrix