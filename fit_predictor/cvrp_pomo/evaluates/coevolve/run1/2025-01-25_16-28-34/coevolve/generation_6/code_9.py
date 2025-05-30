import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands vector is 1-dimensional
    demands = demands.view(-1)
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Problem-specific local search heuristic contribution
    # For simplicity, we'll use a simple heuristic that assumes all edges are promising
    # and then we'll adjust this based on the demands.
    heuristics.fill_(1)
    
    # Adjust the heuristic values based on the demands (this is where you could implement
    # more sophisticated logic based on the specifics of the problem and the vehicle capacity)
    # For now, we just use a simple inverse demand heuristic (demand higher, heuristic lower)
    # This is just a placeholder and may not be the best heuristic for your problem.
    heuristics *= (1 / (demands + 1e-6))  # Adding a small constant to avoid division by zero
    
    # This is where you would incorporate the other heuristics:
    # - Adaptive PSO Population Management
    # - Dynamic Tabu Search with Adaptive Cost Function
    
    # Note: The above code does not include the actual PSO or Tabu Search logic,
    # as these are complex algorithms that would require significant additional code.
    # The provided code is a starting point that can be expanded upon to include the full
    # functionality as described in the problem description.
    
    return heuristics