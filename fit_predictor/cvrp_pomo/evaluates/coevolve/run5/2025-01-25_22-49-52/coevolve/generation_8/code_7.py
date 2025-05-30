import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Normalize demands to be between 0 and 1
    demands = demands / demands.sum()

    # Node partitioning: assign each node to a partition based on demand
    # For simplicity, we use a fixed number of partitions
    num_partitions = 5
    partition_size = (demands.sum() + num_partitions - 1) // num_partitions
    partition_demands = torch.zeros(num_partitions)
    for i in range(num_partitions):
        partition_demands[i] = demands[:partition_size].sum()
        demands = demands[partition_size:]

    # Demand relaxation: reduce the demand of each node in the same partition
    for i in range(num_partitions):
        for j in range(num_partitions):
            if i != j:
                heuristics += (demands - partition_demands[i]) * (partition_demands[j] - partition_demands[i])

    # Path decomposition: encourage using edges that lead to or from high-demand nodes
    for i in range(len(demands)):
        heuristics[i] += demands[i] * (1 + heuristics[i])
        for j in range(len(demands)):
            heuristics[j, i] += demands[j] * (1 + heuristics[j, i])

    # Dynamic window approach: adjust heuristics based on the distance to the nearest partition center
    partition_centers = partition_demands.cumsum()[:-1]
    for i in range(len(demands)):
        nearest_center = partition_centers[torch.argmin(torch.abs(partition_centers - demands[i]))]
        heuristics[i] += 1 / (1 + distance_matrix[i, 0] + distance_matrix[0, i])

    return heuristics