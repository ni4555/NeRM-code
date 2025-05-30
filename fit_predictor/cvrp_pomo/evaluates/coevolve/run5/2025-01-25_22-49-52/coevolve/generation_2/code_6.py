import random
import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands with respect to vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Node partitioning: Partition the nodes into clusters based on demands
    # Here we use a simple threshold-based approach for demonstration
    threshold = 0.5
    high_demand_clusters = torch.where(normalized_demands > threshold)[0]
    low_demand_clusters = torch.where(normalized_demands <= threshold)[0]
    
    # Demand relaxation: Relax the demands slightly to improve load balancing
    relaxed_demands = demands * 0.95
    
    # Path decomposition: Calculate heuristic values for paths within and between clusters
    for i in range(n):
        if i in high_demand_clusters:
            # High demand nodes have higher heuristic values
            heuristic_matrix[i] = -distance_matrix[i]
        else:
            # Low demand nodes have lower heuristic values
            heuristic_matrix[i] = distance_matrix[i]
    
    # Dynamic window technique: Adjust heuristic values based on current vehicle capacities
    # For simplicity, we simulate a dynamic window by considering a random change in capacities
    current_capacities = relaxed_demands
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate the potential load on the path i->j
            potential_load = relaxed_demands[i] + relaxed_demands[j]
            # Adjust heuristic based on capacity constraints
            if potential_load > current_capacities[i]:
                heuristic_matrix[i, j] += 1
            if potential_load > current_capacities[j]:
                heuristic_matrix[j, i] += 1
    
    return heuristic_matrix