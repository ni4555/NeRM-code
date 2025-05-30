import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the total vehicle capacity
    # Assuming that the total vehicle capacity is the sum of all demands
    total_capacity = demands.sum()
    
    # Demand relaxation: Increase the demand of each customer by a small factor
    # This factor is chosen to make the demands larger but still feasible
    demand_factor = 1.1
    relaxed_demands = demands * demand_factor
    
    # Node partitioning: Partition the customers into groups based on their relaxed demands
    # For simplicity, we can use a threshold to create partitions
    threshold = total_capacity / 2
    relaxed_demand_mask = relaxed_demands > threshold
    partitioned_mask = relaxed_demand_mask.type(torch.float32)
    
    # Path decomposition: Assign a higher heuristic value to paths that include more customers
    # from the same partition
    path_value = torch.sum(distance_matrix * partitioned_mask, dim=1)
    
    # Multi-objective evolutionary algorithm approach: Use a simple heuristic to balance
    # the path values with the original distances
    # We can use a weighted sum of the path value and the negative distance
    # to create a balanced heuristic value
    weight = 0.5
    heuristic_values = weight * path_value - (1 - weight) * distance_matrix
    
    # Dynamic window approach: Introduce a dynamic window of time for considering
    # the current and future demand changes
    # For simplicity, we will consider the current demand and the next demand
    # This is a placeholder for a more complex dynamic window implementation
    future_demand = relaxed_demands[1:]
    dynamic_window_mask = torch.cat([relaxed_demand_mask, future_demand > threshold], dim=0)
    dynamic_window_heuristic = weight * path_value - (1 - weight) * distance_matrix * dynamic_window_mask
    
    # Combine the dynamic window heuristic with the path decomposition heuristic
    combined_heuristic = (heuristic_values + dynamic_window_heuristic) / 2
    
    return combined_heuristic