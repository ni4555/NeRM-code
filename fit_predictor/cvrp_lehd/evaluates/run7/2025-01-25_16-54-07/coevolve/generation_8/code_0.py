import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    
    # Create cumulative demand mask
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Create edge feasibility mask based on vehicle capacity
    edge_feasibility = (cumulative_demand - demands[:, None]) <= vehicle_capacity
    
    # Calculate contribution to balanced load distribution
    load_contribution = (vehicle_capacity - (cumulative_demand - demands[:, None])) / vehicle_capacity
    
    # Combine the masks with load contribution
    heuristics = load_contribution * edge_feasibility.float() - (1 - edge_feasibility.float())
    
    return heuristics