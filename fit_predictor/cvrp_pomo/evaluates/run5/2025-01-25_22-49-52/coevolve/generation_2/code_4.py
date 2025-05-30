import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized
    demands = demands / demands.sum()

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()

    # Define the dynamic window technique parameters
    alpha = 0.5  # Dynamic window adjustment factor
    window_size = 0.1  # Window size for dynamic adjustment

    # Node partitioning and demand relaxation
    num_nodes = distance_matrix.size(0)
    node_partition = torch.argsort(demands)[::-1]  # Sort nodes by demand in descending order

    # Decompose the problem into smaller subproblems
    num_partitions = 5  # Number of partitions
    partition_size = (num_nodes + num_partitions - 1) // num_partitions
    partition_demands = torch.zeros_like(demands)
    for i in range(num_partitions):
        partition_demands[i * partition_size:(i + 1) * partition_size] = demands[node_partition[i * partition_size:(i + 1) * partition_size]]

    # Calculate heuristic values using a multi-objective evolutionary algorithm
    # For simplicity, we use a simple heuristic: the inverse of the demand
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Adjust the heuristic value based on the dynamic window technique
                heuristic_matrix[i, j] = (1 / demands[j]) * alpha + (1 - alpha) * (1 / demands[j])
                # Adjust based on partition demand
                if i >= num_partitions * partition_size and j < num_partitions * partition_size:
                    heuristic_matrix[i, j] *= partition_demands[j]
                elif i < num_partitions * partition_size and j >= num_partitions * partition_size:
                    heuristic_matrix[i, j] *= partition_demands[i]

    return heuristic_matrix

# Example usage:
# distance_matrix = torch.tensor([[0, 10, 20, 30], [10, 0, 15, 25], [20, 15, 0, 10], [30, 25, 10, 0]])
# demands = torch.tensor([0.5, 0.2, 0.3, 0.0])
# print(heuristics_v2(distance_matrix, demands))