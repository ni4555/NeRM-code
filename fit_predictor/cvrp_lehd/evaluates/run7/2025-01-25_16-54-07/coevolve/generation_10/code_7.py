import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    
    # Demand normalization
    normalized_demands = demands / vehicle_capacity
    
    # Cumulative demand mask
    cumulative_demand_mask = torch.cumsum(normalized_demands, dim=0)
    
    # Edge feasibility mask
    edge_feasibility_mask = (cumulative_demand_mask < vehicle_capacity).float()
    
    # Clear edge evaluation
    edge_evaluation = -distance_matrix
    
    # Combine all factors for heuristic evaluation
    heuristic_values = (edge_feasibility_mask * edge_evaluation).sum(dim=1)
    
    return heuristic_values