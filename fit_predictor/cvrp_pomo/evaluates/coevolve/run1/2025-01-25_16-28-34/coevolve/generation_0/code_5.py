import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the sum of normalized demands for each edge
    edge_demands = torch.sum(normalized_demands[:, None] * normalized_demands, axis=0)
    
    # Calculate the heuristics as the negative of the sum of normalized demands
    # This assumes that we want to penalize edges with higher demand
    heuristics = -edge_demands
    
    # Ensure the diagonal elements are set to a very low value (not included in the heuristic)
    torch.fill_diagonal_(heuristics, float('-inf'))
    
    return heuristics