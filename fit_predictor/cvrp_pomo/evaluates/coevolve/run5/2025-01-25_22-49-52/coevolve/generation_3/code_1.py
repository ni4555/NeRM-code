import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Node partitioning: Divide nodes into groups based on demand
    partition_threshold = demands.mean()
    partitions = torch.where(demands < partition_threshold, 0, 1)
    
    # Demand relaxation: Relax demands slightly to allow for better load balancing
    relaxed_demands = demands * 0.9
    
    # Path decomposition: Use a heuristic to estimate the cost of paths
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the heuristic for the edge (i, j)
                # This is a simplified example using the distance and relaxed demand
                edge_heuristic = distance_matrix[i, j] - relaxed_demands[i] - relaxed_demands[j]
                
                # Apply node partitioning to the heuristic
                if partitions[i] == partitions[j]:
                    edge_heuristic *= 0.5  # Increase the heuristic for edges within the same partition
                
                # Update the heuristic matrix
                heuristic_matrix[i, j] = edge_heuristic
    
    # Apply dynamic window approach: Adjust the heuristic based on the current state of the problem
    # This is a placeholder for the dynamic window approach logic
    # For example, if a vehicle is approaching its capacity, increase the heuristic for incoming edges
    # For simplicity, we will not implement this part

    return heuristic_matrix