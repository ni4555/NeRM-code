import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check if the distance_matrix and demands have the same number of nodes
    if distance_matrix.shape[0] != distance_matrix.shape[1] or demands.shape[0] != distance_matrix.shape[0]:
        raise ValueError("Distance matrix and demands must have the same number of nodes.")
    
    # Normalize demands by the total vehicle capacity (assuming 1 vehicle for simplicity)
    total_capacity = 1  # This should be replaced with the actual total vehicle capacity if needed
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance heuristic
    heuristic_matrix = -distance_matrix
    
    # Apply capacity-based penalty for edges close to the vehicle's capacity
    # We define 'close' as any edge that is within 0.1 of the vehicle's capacity
    capacity_penalty_threshold = 0.1 * total_capacity
    capacity_penalty = torch.where(normalized_demands < capacity_penalty_threshold, 
                                   1.0, 
                                   2.0)  # Increase the heuristic value for these edges
    
    # Combine the heuristic with the capacity penalty
    combined_heuristic = heuristic_matrix + capacity_penalty
    
    return combined_heuristic

# Example usage:
# Assuming we have a distance matrix and a vector of customer demands
# distance_matrix = torch.tensor([[0, 2, 5, 3], [2, 0, 6, 1], [5, 6, 0, 2], [3, 1, 2, 0]], dtype=torch.float32)
# demands = torch.tensor([10, 15, 8, 5], dtype=torch.float32)
# heuristics_matrix = heuristics_v2(distance_matrix, demands)