import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the load factor for each edge
    load_factor = distance_matrix * demands.unsqueeze(1)
    
    # Normalize the load factor by the total capacity to get the load per edge
    load_per_edge = load_factor / total_capacity
    
    # Calculate the heuristic value for each edge based on load per edge
    # Negative values indicate undesirable edges, positive values indicate promising ones
    heuristic_matrix = -load_per_edge
    
    # Adjust the heuristic values to ensure some edges are more promising than others
    # This can be done by adding a constant to the undesirable edges
    # Here, we add a small positive constant to make the values more distinct
    small_constant = 0.1
    undesirable_edges = load_per_edge.abs() > 0.5
    heuristic_matrix[undesirable_edges] += small_constant
    
    return heuristic_matrix