import random
import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Demand relaxation: Add a small value to the demands to allow for more flexibility
    demand_relaxation = 0.01
    relaxed_demands = demands + demand_relaxation
    
    # Node partitioning: Divide the nodes into partitions based on demand
    # Here we use a simple method of partitioning by demand threshold
    demand_threshold = relaxed_demands.mean()
    partitions = (relaxed_demands > demand_threshold).float()
    
    # Dynamic window technique: Initialize the dynamic window
    dynamic_window = torch.arange(n)
    
    # Multi-objective evolutionary algorithm (MOEA) approach:
    # Here we use a simplified version by evaluating a random subset of edges
    num_evaluations = 10
    evaluations = torch.rand(num_evaluations, n)
    fitness_scores = -evaluations  # Negative fitness for minimization
    
    # Update the heuristic matrix based on the fitness scores
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the heuristic value based on the current edge
                edge_heuristic = -distance_matrix[i, j] - relaxed_demands[i] - relaxed_demands[j]
                # Add the fitness score from the MOEA
                edge_heuristic += fitness_scores[i, j]
                # Update the heuristic matrix
                heuristic_matrix[i, j] = edge_heuristic
    
    # Apply path decomposition to prioritize edges within the same partition
    for partition in torch.unique(partitions):
        partition_mask = partitions == partition
        partition_indices = torch.where(partition_mask)[0]
        partition_distance_matrix = distance_matrix[partition_indices, partition_indices]
        partition_demands = relaxed_demands[partition_indices]
        # Update the heuristic within the partition
        for i in range(len(partition_indices)):
            for j in range(i + 1, len(partition_indices)):
                edge_index = partition_indices[i] * n + partition_indices[j]
                heuristic_matrix[partition_indices[i], partition_indices[j]] = -partition_distance_matrix[i, j] - partition_demands[i] - partition_demands[j]
    
    return heuristic_matrix