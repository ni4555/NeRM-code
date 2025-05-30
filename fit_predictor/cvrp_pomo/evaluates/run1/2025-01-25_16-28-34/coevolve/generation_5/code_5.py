import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be between 0 and 1 based on total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the cost for each edge based on the demand and distance
    # Negative values are used for undesirable edges, and positive values for promising ones
    # A simple heuristic could be to multiply demand by distance, but other functions could be used
    # depending on the specifics of the problem.
    cost_matrix = normalized_demands.unsqueeze(1) * distance_matrix
    
    # The cost matrix can be modified to introduce additional heuristics,
    # for example, adding a penalty for longer distances or for routes that go against
    # a certain direction, if known from the problem context.
    
    # In this example, we do not add any such penalties for simplicity.
    
    return cost_matrix

# Example usage:
# Create a sample distance matrix and demands vector
distance_matrix = torch.tensor([
    [0, 3, 5, 10],
    [2, 0, 2, 7],
    [1, 8, 0, 9],
    [4, 1, 2, 0]
], dtype=torch.float32)

demands = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

# Call the function
heuristics_matrix = heuristics_v2(distance_matrix, demands)
print(heuristics_matrix)