import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demand vector is of shape (n,)
    demands = demands.unsqueeze(0).expand_as(distance_matrix)
    
    # Normalize demands by the total vehicle capacity (assuming a single vehicle for simplicity)
    vehicle_capacity = demands.sum().item()
    normalized_demands = demands / vehicle_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Node partitioning: Create two partitions based on a threshold value
    # This is a simplistic approach, real-world implementation might be more complex
    threshold = 0.5
    partition1 = normalized_demands < threshold
    partition2 = normalized_demands >= threshold
    
    # Calculate relative distances between partitions
    relative_distances = distance_matrix[partition1][:, partition2] - distance_matrix[partition2][:, partition1]
    
    # Demand relaxation: Adjust the heuristic values based on the difference in demands
    adjusted_heuristics = heuristics + torch.where(partition1, -relative_distances, 0)
    adjusted_heuristics += torch.where(partition2, relative_distances, 0)
    
    # Multi-objective evolutionary algorithm component
    # This is a placeholder for the actual evolutionary algorithm logic
    # For simplicity, we'll assume that the adjusted heuristic values are already good enough
    # In a real-world scenario, this would be replaced with a call to an evolutionary algorithm
    # and the heuristics matrix would be modified accordingly
    
    # Return the heuristics matrix
    return adjusted_heuristics