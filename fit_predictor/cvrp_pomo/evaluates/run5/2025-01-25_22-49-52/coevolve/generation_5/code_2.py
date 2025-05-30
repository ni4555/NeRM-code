import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the sum of demands to get a capacity ratio
    demand_ratio = demands / demands.sum()
    
    # Calculate the distance-based potential
    distance_potential = -distance_matrix
    
    # Calculate the demand-based potential
    demand_potential = -torch.sum(demand_ratio.unsqueeze(0) * distance_matrix.unsqueeze(1), dim=2)
    
    # Combine the two potentials to get the total potential
    total_potential = distance_potential + demand_potential
    
    # Normalize the potential to ensure consistent scaling
    max_potential = torch.max(total_potential)
    min_potential = torch.min(total_potential)
    normalized_potential = (total_potential - min_potential) / (max_potential - min_potential)
    
    return normalized_potential