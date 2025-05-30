import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    total_capacity = demands.sum()
    demand_vector = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics based on demand vector and distance matrix
    # We can use the negative of the inverse of the distance to encourage short paths
    heuristics_matrix = -torch.log(distance_matrix)
    
    # Incorporate demand into the heuristic value
    heuristics_matrix = heuristics_matrix * demand_vector.unsqueeze(0)
    
    # Apply some heuristic like the triangle inequality to avoid suboptimal solutions
    # Here we're just an example of adding a simple threshold based on the demand
    heuristics_matrix[distance_matrix > 1.5] += -1
    
    return heuristics_matrix