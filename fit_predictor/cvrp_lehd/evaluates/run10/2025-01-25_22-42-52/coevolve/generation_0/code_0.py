import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the sum of normalized demands for each edge
    edge_demands = torch.sum(normalized_demands[:, None] * normalized_demands, dim=0)
    
    # Calculate the edge heuristics as the negative of the sum of normalized demands
    # since negative values are desirable here
    heuristics = -edge_demands
    
    return heuristics