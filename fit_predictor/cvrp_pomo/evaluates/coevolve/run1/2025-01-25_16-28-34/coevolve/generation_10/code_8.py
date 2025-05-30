import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands to represent the fraction of the total capacity each customer requires
    normalized_demands = demands / total_capacity
    
    # Calculate the demand contribution for each edge
    # We use a simple heuristic where the demand contribution is the normalized demand of the destination node
    demand_contributions = normalized_demands[1:]  # Exclude the depot node (index 0)
    
    # Calculate the distance contribution for each edge
    # Here we use the distance to the next customer as a proxy for the edge contribution
    # For the last customer, we set the distance contribution to 0 since it's the end of the route
    distance_contributions = distance_matrix[1:, 1:]  # Exclude the diagonal and the depot node
    distance_contributions = distance_contributions.fill_diagonal_(0)
    
    # Combine the demand and distance contributions to get the heuristic values
    # We use a simple linear combination where the weight for demand is 0.5 and for distance is 0.5
    # This can be adjusted based on the specific problem characteristics
    heuristic_values = 0.5 * demand_contributions + 0.5 * distance_contributions
    
    # Return the heuristic matrix
    return heuristic_values