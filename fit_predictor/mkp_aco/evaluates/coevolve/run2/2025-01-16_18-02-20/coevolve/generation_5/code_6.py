import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # This is a placeholder for the actual implementation.
    # It should return an array of heuristic values of shape (n,)
    # where n is the number of items.
    # For this example, let's assume a simple heuristic based on value-to-weight ratio.
    return prize / (weight + 1e-10)  # Adding a small value to avoid division by zero

def solve_mkp(prize: np.ndarray, weight: np.ndarray, num_knapsacks: int) -> np.ndarray:
    # Ensure that the prize and weight arrays are 1D
    prize = prize.flatten()
    weight = weight.flatten()
    
    n = len(prize)  # Number of items
    m = weight.shape[1]  # Number of knapsacks (assuming all weights are 1, m is the number of dimensions)
    
    # Ensure that the number of knapsacks is consistent with the weight dimensions
    if m != 1:
        raise ValueError("Expected 1D weight array for MKP with fixed weight constraints of 1.")
    
    # Calculate initial heuristic values
    heuristic_values = heuristics_v2(prize, weight)
    
    # Initialize a list to store selected items for each knapsack
    selected_items = [[] for _ in range(num_knapsacks)]
    
    # Iterate over the items to populate the knapsacks
    for _ in range(n):
        # Select the item with the highest heuristic value
        item_to_select = np.argmax(heuristic_values)
        
        # Check if the item can be added to any knapsack
        for knapsack in range(num_knapsacks):
            if sum([item_weight for item in selected_items[knapsack] for item_weight in weight[item_to_select]]) < 1:
                selected_items[knapsack].append(item_to_select)
                break
        
        # Update the heuristic values to reflect the new state of the knapsacks
        heuristic_values = heuristics_v2(prize, weight)
    
    # Calculate the total prize collected
    total_prize = sum(prize[item] for item in selected_items)
    
    return selected_items, total_prize

# Example usage:
# prize = np.array([60, 100, 120])
# weight = np.array([[1], [1], [1]])  # 1D array of weights, all 1s
# num_knapsacks = 2
# selected_items, total_prize = solve_mkp(prize, weight, num_knapsacks)
# print("Selected items:", selected_items)
# print("Total prize:", total_prize)