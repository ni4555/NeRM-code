import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand-to-distance ratio
    demand_to_distance = demands / distance_matrix
    
    # Normalize the demand-to-distance ratio by the maximum value in each row
    # This helps in promoting edges with lower ratios (i.e., more efficient routes)
    row_max = torch.max(demand_to_distance, dim=1)[0]
    normalized_demand_to_distance = demand_to_distance / row_max[:, None]
    
    # Apply a negative heuristic for edges with high demand-to-distance ratio
    # We use a negative value to discourage these edges in the heuristic evaluation
    heuristics = -1 * (1 - normalized_demand_to_distance)
    
    return heuristics