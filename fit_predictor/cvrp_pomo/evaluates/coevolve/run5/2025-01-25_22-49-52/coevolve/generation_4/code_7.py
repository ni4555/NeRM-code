import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize node distances
    distance_matrix = torch.clamp(distance_matrix / distance_matrix.max(), min=0.0, max=1.0)
    
    # Normalize demands
    total_demand = demands.sum()
    demands = demands / total_demand
    
    # Calculate potential values for explicit depot handling
    depot_potential = distance_matrix.sum() * demands.sum()
    
    # Create an initial heuristic value matrix
    heuristic_values = distance_matrix * demands
    
    # Incorporate normalization of potential values
    heuristic_values = heuristic_values / depot_potential
    
    # Apply a simple dynamic window approach by considering only edges that are within a certain factor of the shortest distance
    # This factor can be tuned to adjust the trade-off between the quality of the heuristic and its efficiency
    shortest_distances = torch.min(distance_matrix, dim=1, keepdim=True)[0]
    factor = 1.5
    heuristic_values[distance_matrix > shortest_distances * factor] = 0.0
    
    return heuristic_values