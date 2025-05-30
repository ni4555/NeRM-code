import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming that the demands vector includes the demand of the depot (index 0)
    # which is 0 because the depot is not a customer.
    
    # Calculate the total demand to normalize
    total_demand = demands.sum()
    
    # Normalize the demands vector
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics using a simple heuristic that takes into account the
    # distance and the normalized demand. A simple approach is to use a negative
    # weight for distance and a positive weight for demand.
    # Here, we are using a linear combination where the coefficients can be adjusted
    # according to the problem's specifics.
    # Negative coefficients for distance (to discourage longer distances)
    # Positive coefficients for demand (to encourage visiting customers with higher demands)
    alpha_distance = -1.0
    alpha_demand = 1.0
    
    # Create a tensor of all ones to represent the distance to the depot (which is 0)
    # and then subtract the distance to the depot for each customer.
    # This creates a tensor with the distance from the depot to each customer.
    distance_from_depot = distance_matrix[:, 1:] - distance_matrix[1:, :]
    
    # Use the negative of the distance to create a weight for the distance.
    distance_weight = alpha_distance * (-distance_from_depot)
    
    # Multiply the distance weight by the normalized demand to get the heuristic value for each edge.
    heuristics = distance_weight * normalized_demands
    
    # Add the demand at the depot as a zero value to align the heuristics tensor shape with the distance matrix.
    heuristics = torch.cat((torch.zeros(1), heuristics))
    
    return heuristics