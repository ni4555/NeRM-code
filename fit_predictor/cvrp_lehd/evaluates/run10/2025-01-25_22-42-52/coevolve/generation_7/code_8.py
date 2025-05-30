import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to get the fraction of capacity required by each customer
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost for each edge
    # This is a simple heuristic that assumes the cost is proportional to the demand
    potential_costs = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Calculate the distance-based cost for each edge
    distance_costs = distance_matrix
    
    # Combine the demand-based and distance-based costs
    combined_costs = potential_costs + distance_costs
    
    # Introduce a penalty for undesirable edges (e.g., very high demand or long distance)
    # Here we use a simple heuristic where we add a large negative value for high demand
    # and a smaller negative value for high distance
    penalty = -torch.where(potential_costs > 1, potential_costs - 1, -0.1 * distance_costs)
    
    # Subtract the penalty from the combined costs to get the heuristic values
    heuristics = combined_costs - penalty
    
    return heuristics