import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    total_capacity = demands[0]  # Assuming the first element is the vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Normalize distances
    min_distance = distance_matrix.min()
    max_distance = distance_matrix.max()
    normalized_distances = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Calculate potential values for explicit depot handling
    depot_potential = torch.ones_like(normalized_distances) * (normalized_distances.max() - normalized_distances)
    
    # Calculate the demand-based potential
    demand_potential = normalized_distances * normalized_demands
    
    # Combine the potentials into the heuristic
    heuristic = (depot_potential + demand_potential).unsqueeze(1)  # Add an extra dimension for broadcasting
    
    # Subtract the heuristic values to create a promise for each edge
    # Negative values for undesirable edges, positive for promising ones
    edge_potentials = -1 * (heuristic.sum(dim=0) - 1)  # Sum across the demand dimension
    
    return edge_potentials