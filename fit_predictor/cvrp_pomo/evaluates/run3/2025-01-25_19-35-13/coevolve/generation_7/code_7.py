import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the sum of normalized demands for each edge
    edge_demand_sum = torch.matmul(normalized_demands.unsqueeze(1), normalized_demands.unsqueeze(0))

    # Incorporate distance and road quality factors (assuming road_quality_matrix is available)
    # For the sake of this example, let's assume road_quality_matrix is a 1-hot encoded matrix with shape n by n
    # road_quality_matrix = ... (not provided in the question, so we'll skip this part)

    # Calculate the potential function by combining edge_demand_sum and distance
    # For simplicity, we'll just use distance_matrix directly, as road_quality_matrix is not provided
    potential_function = edge_demand_sum - distance_matrix

    # Refine the potential function to prevent division by zero errors
    potential_function = torch.clamp(potential_function, min=-float('inf'), max=float('inf'))

    return potential_function