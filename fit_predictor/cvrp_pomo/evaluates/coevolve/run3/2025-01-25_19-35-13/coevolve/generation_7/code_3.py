import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands relative to vehicle capacity (assuming total capacity is 1 for simplicity)
    normalized_demands = demands / demands.sum()
    
    # Calculate the sum of normalized demands for each edge
    edge_demand_sum = torch.matmul(normalized_demands, normalized_demands.t())
    
    # Incorporate distance and road quality factors (assuming these are given as multipliers)
    distance_multiplier = torch.from_numpy(distance_matrix).to(torch.float32)
    road_quality_multiplier = torch.ones_like(distance_multiplier)  # Assuming equal road quality for simplicity
    
    # Combine the demand sum with distance and road quality
    combined_potential = edge_demand_sum * distance_multiplier * road_quality_multiplier
    
    # Refine the potential function to prevent division by zero errors
    combined_potential = torch.clamp(combined_potential, min=1e-10)
    
    # Invert the potential function to have negative values for undesirable edges
    heuristics = -combined_potential
    
    return heuristics