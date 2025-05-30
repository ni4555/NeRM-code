import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Calculate total demand
    total_demand = demands.sum()
    
    # Demand relaxation based on total capacity
    relaxed_demands = (demands / total_demand) * (n - 1)
    
    # Node partitioning: Assign each node to a partition based on its demand
    # We will use a simple approach for demonstration purposes: assign nodes with demand greater than a threshold to a different partition
    threshold = 0.1  # Threshold for partitioning, 10% of total demand
    high_demand_mask = relaxed_demands > threshold
    low_demand_mask = ~high_demand_mask
    
    # Calculate distance penalties for high and low demand nodes
    high_demand_penalty = distance_matrix[high_demand_mask].mean()
    low_demand_penalty = distance_matrix[low_demand_mask].mean()
    
    # Apply penalties based on partitioning
    high_demand_distance_penalty = torch.where(high_demand_mask.unsqueeze(1), distance_matrix, high_demand_penalty)
    low_demand_distance_penalty = torch.where(low_demand_mask.unsqueeze(1), distance_matrix, low_demand_penalty)
    
    # Path decomposition: Assign weights to edges based on distance penalties and demand relaxation
    path_weights = high_demand_distance_penalty + low_demand_distance_penalty - (relaxed_demands.unsqueeze(1) + relaxed_demands)
    
    # Normalize path weights to create heuristics
    heuristics = path_weights / path_weights.sum()
    
    # Ensure the heuristics are in the desired range by adding a constant
    heuristics += (1 - heuristics.max())
    
    return heuristics