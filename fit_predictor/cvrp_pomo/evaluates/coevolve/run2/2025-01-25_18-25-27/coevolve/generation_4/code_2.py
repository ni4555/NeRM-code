import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Assuming demands are normalized by the total vehicle capacity, we can use them directly.
    
    # Calculate the sum of all demands to normalize the distance matrix.
    total_demand = demands.sum()
    
    # Calculate the demand contribution for each edge.
    # This is the product of the distances and the ratio of the demands of the two nodes.
    edge_demand_contributions = (distance_matrix * demands.unsqueeze(1) * demands.unsqueeze(0)).sum(dim=2)
    
    # Normalize the distance matrix by the sum of all demands to get the relative costs.
    normalized_distance_matrix = distance_matrix / total_demand
    
    # Calculate the heuristic values based on the normalized distances and demand contributions.
    # Negative values for undesirable edges and positive values for promising ones.
    heuristics = normalized_distance_matrix - edge_demand_contributions
    
    return heuristics