import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming distance_matrix and demands are both of shape (n, n) and (n,)
    # where n is the number of nodes (including the depot node indexed by 0)
    
    # Calculate the load factor for each edge
    # The load factor is a measure of the potential load if a vehicle travels this edge
    load_factor = demands.unsqueeze(0) * demands.unsqueeze(1)
    
    # Calculate the negative distance factor for each edge
    # The idea is that closer nodes are more desirable to visit first (assuming lower travel costs)
    negative_distance_factor = -distance_matrix
    
    # Combine the load factor and the negative distance factor
    combined_factor = load_factor + negative_distance_factor
    
    # Normalize the combined factor by dividing by the maximum possible load (vehicle capacity)
    # and scaling to a suitable range for the heuristic function
    max_demand = demands.max()
    normalized_combined_factor = combined_factor / max_demand
    
    # Introduce a small negative constant to ensure all edge weights are negative (undesirable)
    # or positive (promising)
    epsilon = torch.tensor(-0.01)
    heuristics = normalized_combined_factor + epsilon
    
    return heuristics