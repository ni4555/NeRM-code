import random
import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are of the correct data type
    distance_matrix = distance_matrix.type(torch.float32)
    demands = demands.type(torch.float32)
    
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with high values for undesirable edges
    heuristic_matrix = torch.full_like(distance_matrix, fill_value=1e9)
    
    # Apply node partitioning to identify clusters of nodes
    # This is a placeholder for an actual node partitioning algorithm
    # For the sake of this example, we'll assume a fixed number of partitions
    num_partitions = 3
    partition_size = len(distance_matrix) // num_partitions
    for start in range(0, len(distance_matrix), partition_size):
        end = min(start + partition_size, len(distance_matrix))
        # Compute some heuristic for the partition
        partition_heuristic = torch.mean(distance_matrix[start:end, start:end])
        heuristic_matrix[start:end, start:end] -= partition_heuristic
    
    # Demand relaxation: reduce the demand slightly to allow for more flexibility
    relaxed_demands = normalized_demands * 0.9
    
    # Path decomposition: create a heuristic based on the relaxed demands
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                # Calculate the combined demand of the path from i to j
                combined_demand = relaxed_demands[i] + relaxed_demands[j]
                # Adjust the heuristic for the edge based on the combined demand
                if combined_demand <= 1.0:  # Only consider paths that fit within the vehicle capacity
                    heuristic_matrix[i, j] = -torch.log(combined_demand)
    
    # Integrate multi-objective evolutionary algorithm approach (simplified here)
    # This would involve creating a population of routes and evolving them
    # For the sake of this example, we'll just add a random component
    random_factor = torch.rand_like(distance_matrix)
    heuristic_matrix -= random_factor * 1e6  # Negative values are desirable
    
    return heuristic_matrix