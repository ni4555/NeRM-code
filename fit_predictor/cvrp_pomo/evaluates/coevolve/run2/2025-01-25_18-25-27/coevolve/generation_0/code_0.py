import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for each row (customer node)
    demand_sums = demands.sum(dim=1, keepdim=True)
    
    # Calculate the maximum demand that can be visited before the vehicle returns
    max_demand = demands / demand_sums * demands.max()
    
    # Calculate the potential of each edge (negative for high demand edges, positive otherwise)
    edge_potential = -distance_matrix * max_demand
    
    # Avoid the negative infinity by setting them to a very small positive value
    edge_potential = torch.clamp(edge_potential, min=1e-10)
    
    # Normalize the potential to be between 0 and 1
    edge_potential = torch.exp(edge_potential)
    edge_potential /= edge_potential.sum()
    
    return edge_potential