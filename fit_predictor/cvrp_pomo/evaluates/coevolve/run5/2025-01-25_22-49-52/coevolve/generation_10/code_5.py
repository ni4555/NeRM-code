import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands = demands / demands.sum()  # Normalize demands by total demand

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Step 1: Node partitioning to optimize path decomposition
    # For simplicity, we'll use a basic partitioning approach where we split the customers into two groups
    # and assign them to different vehicles. This is a heuristic and may not always be optimal.
    partition_threshold = 0.5
    partition_indices = (demands > partition_threshold).nonzero(as_tuple=False).view(-1)
    for i in range(partition_indices.shape[0]):
        if i % 2 == 0:  # Assign even indices to one vehicle
            heuristic_matrix[partition_indices[i], :] = -1
            heuristic_matrix[:, partition_indices[i]] = -1
        else:  # Assign odd indices to another vehicle
            heuristic_matrix[partition_indices[i], :] = -1
            heuristic_matrix[:, partition_indices[i]] = -1

    # Step 2: Demand relaxation to manage dynamic changes
    # We introduce a relaxation factor to allow for some overcapacity in the initial heuristic
    relaxation_factor = 0.1
    for i in range(n):
        for j in range(n):
            if i != j and (i in partition_indices or j in partition_indices):
                # Relax the heuristic for edges between different partitions
                heuristic_matrix[i, j] = -1 + relaxation_factor

    # Step 3: Dynamic window approach to handle real-time changes
    # We assume that the distance matrix can be updated in real-time and recompute the heuristic
    # This step is abstracted here as it would require a real-time update mechanism
    # For the sake of this example, we'll assume the distance matrix is static

    # Step 4: Multi-objective evolutionary algorithm to ensure robustness and adaptability
    # This step is also abstracted, as it would involve a complex evolutionary algorithm
    # For the sake of this example, we'll assume the heuristic is already optimized

    return heuristic_matrix