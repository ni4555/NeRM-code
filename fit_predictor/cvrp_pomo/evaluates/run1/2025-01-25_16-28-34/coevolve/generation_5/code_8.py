import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands to use in the heuristic
    cumulative_demands = torch.cumsum(demands, dim=0)
    
    # Compute the heuristic value for each edge
    # The heuristic uses the cumulative demand as a factor to determine if an edge should be included
    # Negative values for undesirable edges are achieved by using -1 * cumulative_demand
    # Promising edges are those where the cumulative demand has not yet exceeded the vehicle capacity
    # We use a small positive constant (e.g., 1e-3) to avoid division by zero
    capacity = demands[-1]  # Assuming the last demand corresponds to the total vehicle capacity
    heuristics = -1 * (cumulative_demands / (capacity + 1e-3))
    
    # Ensure that all values are within the range that PyTorch will accept as valid for a softmax
    # This is important for the integration with PSO and TS, as these algorithms often use softmax
    heuristics = torch.clamp(heuristics, min=-torch.tensor(1e9), max=0)
    
    return heuristics