import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with large negative values
    heuristic_matrix = -torch.ones_like(distance_matrix)
    
    # Demand relaxation factor
    demand_relaxation_factor = 0.1
    
    # Normalize demands and adjust for relaxation
    normalized_demands = demands / demands.sum()
    relaxed_demands = normalized_demands * (1 + demand_relaxation_factor)
    
    # Node partitioning for path decomposition
    # A simple approach could be to partition nodes based on distance from the depot
    # Here, we'll use a simple threshold to simulate node partitioning
    partition_threshold = n * 0.2  # 20% of the nodes are considered in the first partition
    partition1 = torch.arange(n)[:partition_threshold]
    partition2 = torch.arange(n)[partition_threshold:]
    
    # Calculate heuristic values for edges between partitions
    for i in partition1:
        for j in partition2:
            heuristic_matrix[i, j] = -distance_matrix[i, j] * relaxed_demands[i] * relaxed_demands[j]
    
    # Incorporate dynamic window approach by considering only edges that are not too far
    # from the current vehicle capacity
    # For simplicity, we'll use a dynamic window based on the average demand per vehicle
    average_demand_per_vehicle = demands.sum() / (n - 1)
    dynamic_window_threshold = average_demand_per_vehicle * 0.5
    
    # Adjust the heuristic matrix based on the dynamic window
    for i in range(n):
        for j in range(n):
            if i != j:
                if relaxed_demands[i] + relaxed_demands[j] <= dynamic_window_threshold:
                    # Increase the heuristic value for edges within the dynamic window
                    heuristic_matrix[i, j] = max(heuristic_matrix[i, j], -distance_matrix[i, j])
    
    return heuristic_matrix