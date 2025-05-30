import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands vector is normalized by dividing by the total demand
    total_demand = demands.sum().item()
    demands = demands / total_demand

    # Calculate the initial heuristic values as a function of demands
    # We use the negative of the demand as a heuristic for the initial implementation
    # This encourages the PSO to visit nodes with higher demand earlier
    heuristics = -demands

    # Optionally, you could add additional terms to the heuristic function
    # For example, you could add the distance to the depot (if not the first node)
    # or another term that might encourage certain types of solutions.

    return heuristics