import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative weighted distance by multiplying distance by demand
    # Normalize the demands to sum to 1 for each vehicle
    demand_normalized = demands / demands.sum()
    
    # Create a matrix where each cell represents the weighted distance
    # We use negative values for the distance matrix to reflect priority
    negative_weighted_distance = -distance_matrix * demand_normalized
    
    # Integrate dynamic load balancing by ensuring that no vehicle is overloaded
    # This is a simplified approach and does not guarantee an optimal load balance
    # It simply ensures that the sum of demands for each vehicle does not exceed capacity
    vehicle_capacity = demands.sum()  # Assuming total capacity is equal to total demand
    negative_weighted_distance += (vehicle_capacity - demands)[:, None]  # Add penalty for exceeding capacity
    
    # Proximity-based route planning by giving priority to edges that are closer
    # This is done by adding a penalty for edges that are not in the top-k closest edges
    # For simplicity, we use the k-nearest neighbors approach here
    k = 5  # Number of closest edges to consider
    _, top_k_indices = torch.topk(negative_weighted_distance, k, dim=1, largest=False)
    top_k_edges = negative_weighted_distance.gather(1, top_k_indices)
    penalty = (negative_weighted_distance - top_k_edges).abs()
    negative_weighted_distance += penalty
    
    return negative_weighted_distance