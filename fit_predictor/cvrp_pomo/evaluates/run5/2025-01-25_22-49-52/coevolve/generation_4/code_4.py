import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Normalize the distance matrix by dividing by the maximum distance in the matrix
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate potential values for explicit depot handling
    # This could be based on some heuristic or simple formula
    depot_potential = normalized_distance_matrix[0] * 0.5  # Example: half the maximum distance to depot
    
    # Calculate potential values for each edge
    # Here we use a simple heuristic that combines normalized distance and normalized demand
    edge_potentials = normalized_distance_matrix + normalized_demands
    
    # Apply the depot potential to the edges leading from the depot
    edge_potentials[0] += depot_potential
    
    # Apply a simple constraint programming approach: minimize edges with high demands
    # This could be a simple threshold, or a more complex model depending on the needs
    high_demand_threshold = 1.5  # Example threshold
    edge_potentials[demands > high_demand_threshold] *= -1  # Mark as undesirable
    
    # Apply dynamic window approach: adjust potential values based on current vehicle loads
    # This could be a simple function or a more complex model depending on the needs
    # For simplicity, we assume vehicle loads are balanced and do not adjust the potential values
    
    return edge_potentials