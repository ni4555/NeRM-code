import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are already normalized by the total vehicle capacity
    n = distance_matrix.shape[0]
    
    # Calculate the heuristic value for each edge as the negative of the demand
    # This encourages the GA to prioritize edges with lower demands
    heuristics = -demands[1:]  # Exclude the depot demand (0-indexed)
    
    # Normalize the heuristics by the maximum demand to ensure all values are in a comparable range
    max_demand = torch.max(torch.abs(demands[1:]))  # Exclude the depot demand
    heuristics = heuristics / max_demand
    
    # Create a diagonal matrix with the normalized demands to penalize visiting the same customer twice
    # We set the diagonal to -1 to indicate that revisiting is undesirable
    demand_penalty = torch.eye(n, dtype=torch.float32) * -1
    demand_penalty[0, 0] = 0  # Set the depot diagonal to 0
    
    # Combine the heuristics with the demand penalty
    combined_heuristics = heuristics + demand_penalty
    
    # Optionally, you can further refine the heuristics by considering the distance to the depot
    # or other factors that may influence the solution quality.
    
    return combined_heuristics