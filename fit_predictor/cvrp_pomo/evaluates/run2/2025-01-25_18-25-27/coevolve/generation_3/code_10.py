import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (which is the sum of demands)
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the potential negative impact of an edge (sum of demands of nodes at both ends)
    # We multiply by -1 to invert the sign so that smaller values (less capacity) are considered better
    negative_impact = (distance_matrix * demands.unsqueeze(1) + demands.unsqueeze(0) * distance_matrix) * -1
    
    # Calculate the potential positive impact based on normalized demands
    # We subtract the negative impact from the distance to make shorter distances more positive
    positive_impact = distance_matrix - negative_impact
    
    # Combine the negative and positive impacts
    heuristics = positive_impact + negative_impact
    
    # Normalize the heuristics by the total capacity to ensure the sum of heuristics is the capacity
    heuristics = heuristics / total_capacity
    
    return heuristics