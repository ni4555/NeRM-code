import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize node distances
    distance_matrix = distance_matrix / distance_matrix.max()
    
    # Normalize demands
    demand_sum = demands.sum()
    demands = demands / demand_sum
    
    # Calculate potential values for explicit depot handling
    depot_potential = demands.sum() * distance_matrix[0, :].sum()
    
    # Calculate edge potential values
    edge_potentials = distance_matrix * demands.unsqueeze(1) * demands.unsqueeze(0)
    
    # Add depot potential to all edges to encourage visiting the depot
    edge_potentials += depot_potential
    
    # Invert the potential to make negative values undesirable and positive values promising
    edge_potentials = -edge_potentials
    
    return edge_potentials