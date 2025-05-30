import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    max_demand = demands.max()
    cumulative_demand = torch.cumsum(demands, dim=0)
    edge_potential = -distance_matrix

    # Calculate cumulative demand mask
    cumulative_demand_mask = (cumulative_demand <= total_capacity) * (cumulative_demand <= max_demand)

    # Calculate edge feasibility mask
    edge_feasibility_mask = (cumulative_demand + demands[1:] <= total_capacity)

    # Combine the masks to prioritize edges
    combined_mask = cumulative_demand_mask * edge_feasibility_mask

    # Adjust the potential based on the combined mask
    edge_potential *= combined_mask

    return edge_potential