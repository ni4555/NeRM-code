import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the savings for each edge (i, j) where i is not the depot
    # Savings = demand at i + demand at j - vehicle capacity
    savings_matrix = demands.unsqueeze(1) + demands.unsqueeze(0) - demands[0]
    
    # Calculate the edge costs, which are the distances between nodes
    edge_costs = distance_matrix
    
    # Combine savings and edge costs to get the heuristic values
    # We use the formula: heuristic_value = savings - cost
    # Negative values indicate undesirable edges, positive values indicate promising edges
    heuristic_values = savings_matrix - edge_costs
    
    # Replace negative values with zeros to avoid considering them in the solution
    heuristic_values[heuristic_values < 0] = 0
    
    return heuristic_values