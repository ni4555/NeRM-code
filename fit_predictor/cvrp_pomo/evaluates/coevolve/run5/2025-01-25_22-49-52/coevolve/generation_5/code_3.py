import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate distance-based potential
    distance_potential = -distance_matrix
    
    # Calculate demand-based potential
    demand_potential = -normalized_demands * demands
    
    # Combine distance and demand potentials
    combined_potential = distance_potential + demand_potential
    
    # Apply normalization to ensure consistent scaling
    max_potential = combined_potential.max()
    min_potential = combined_potential.min()
    normalized_potential = (combined_potential - min_potential) / (max_potential - min_potential)
    
    return normalized_potential