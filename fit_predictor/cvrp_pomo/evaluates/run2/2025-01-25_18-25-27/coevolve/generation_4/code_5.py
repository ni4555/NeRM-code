import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are 1-dimensional
    demands = demands.view(-1)
    
    # Calculate the heuristic value for each edge
    # This heuristic assumes that the cost of an edge is a product of its distance and the demand at the destination node
    # Edges with higher demand and distance product will have higher heuristic values (more promising)
    # This is a simple heuristic and may need to be adjusted based on the specific problem details
    heuristic_values = distance_matrix * demands
    
    return heuristic_values

# Example usage:
# Assuming we have a distance matrix and a demand vector
distance_matrix = torch.tensor([[0, 2, 4, 3], [2, 0, 5, 6], [4, 5, 0, 1], [3, 6, 1, 0]])
demands = torch.tensor([1, 2, 1, 1])

# Calculate the heuristics
promising_edges = heuristics_v2(distance_matrix, demands)
print(promising_edges)