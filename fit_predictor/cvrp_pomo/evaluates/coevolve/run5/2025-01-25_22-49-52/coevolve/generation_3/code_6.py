import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check if the demands are normalized
    if not torch.isclose(demands.sum(), 1.0):
        raise ValueError("Demands must be normalized by the total vehicle capacity.")

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Node partitioning: Create partitions based on demand
    num_nodes = distance_matrix.shape[0]
    partition_threshold = 0.5  # Example threshold for partitioning
    demand_thresholds = torch.quantile(demands, torch.linspace(0, 1, 10))

    # Demand relaxation: Relax demands slightly to improve heuristic performance
    relaxed_demands = demands * 0.95

    # Path decomposition: Calculate heuristics for each node pair
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Calculate the heuristic based on distance and demand
                heuristic = distance_matrix[i, j] * (1 - relaxed_demands[i]) * (1 - relaxed_demands[j])
                
                # Apply dynamic window approach: Adjust heuristics based on current vehicle capacity
                if i == 0:  # Assuming the first node is the depot
                    heuristic *= (1 - demands[j])
                
                # Add the heuristic to the matrix
                heuristic_matrix[i, j] = heuristic

    return heuristic_matrix