import torch
import numpy as np
from typing import Tuple, List

class TabuSearch:
    def __init__(self,
                 prize,      # shape [n,]
                 weight,     # shape [m, n]
                 heuristic,  # not used in tabu search
                 n_solutions=30,  # number of initial solutions
                 tabu_tenure=7,   # length of tabu list
                 device='cpu'
                 ):
        # Ensure prize is on the correct device and has the right shape
        self.prize = torch.as_tensor(prize, device=device)
        if len(self.prize.shape) == 0:
            self.prize = self.prize.unsqueeze(0)
        
        # Ensure weight is on the correct device and has correct shape
        self.weight = torch.as_tensor(weight, device=device)
        # If weight shape doesn't match prize, try transposing it
        if self.weight.shape[1] != self.prize.shape[0]:
            self.weight = self.weight.T
        
        self.m, self.n = self.weight.shape
        
        # Verify dimensions match
        if self.prize.shape[0] != self.n:
            raise ValueError(f"Prize shape {self.prize.shape[0]} must match weight shape[1] {self.n}")
        
        self.n_solutions = n_solutions
        self.tabu_tenure = tabu_tenure
        self.device = device
        
        self.tabu_list = []
        self.best_solution = None
        self.best_obj = 0
        
    def is_feasible(self, solution: torch.Tensor) -> bool:
        """Check if a solution satisfies all constraints"""
        # weight shape is [m, n], solution shape is [n]
        # Need to multiply and sum along n dimension
        weights = (solution * self.weight).sum(dim=1)  # [m,]
        return (weights <= 1).all()
    
    def objective(self, solution: torch.Tensor) -> float:
        """Calculate objective value for a solution"""
        return (solution * self.prize).sum()
    
    def generate_initial_solution(self) -> torch.Tensor:
        """Generate a solution using enhanced efficiency-based selection"""
        solution = torch.zeros(self.n, device=self.device)
        
        # Calculate efficiency with weighted components
        total_weights = self.weight.sum(dim=0)  # [n,]
        max_weights = self.weight.max(dim=0)[0]  # [n,]
        avg_weights = self.weight.mean(dim=0)  # [n,]
        efficiency = self.prize / (0.4 * total_weights + 0.4 * max_weights + 0.2 * avg_weights + 1e-10)
        
        # Sort by efficiency with dynamic randomness
        rand_factor = 0.1 * (1 - torch.arange(self.n, device=self.device) / self.n)
        indices = torch.argsort(efficiency + rand_factor * torch.rand_like(efficiency), descending=True)
        
        for idx in indices:
            # Dynamic probability calculation
            prob = min(0.7, (efficiency[idx] / efficiency[indices[0]]) ** 0.8)
            if torch.rand(1) < prob:
                temp_sol = solution.clone()
                temp_sol[idx] = 1
                if self.is_feasible(temp_sol):
                    solution = temp_sol
        
        return solution
    
    def get_neighborhood(self, solution: torch.Tensor) -> List[Tuple[torch.Tensor, List[int]]]:
        """Generate neighborhood using 1-flip and moderate 2-flip moves"""
        neighbors = []
        
        # 1-flip neighbors
        zeros = torch.where(solution == 0)[0]
        ones = torch.where(solution == 1)[0]
        
        # Try adding items (flip 0 to 1)
        for idx in zeros[:30]:  # Increased from 20 to 30
            new_sol = solution.clone()
            new_sol[idx] = 1
            if self.is_feasible(new_sol):
                neighbors.append((new_sol, [idx.item()]))
        
        # Try removing items (flip 1 to 0)
        for idx in ones:
            new_sol = solution.clone()
            new_sol[idx] = 0
            neighbors.append((new_sol, [idx.item()]))
        
        # 2-flip neighbors
        if len(ones) > 0 and len(zeros) > 0:
            n_exchanges = min(8, len(ones))  # Increased from 5 to 8
            for _ in range(n_exchanges):
                remove_idx = ones[torch.randint(len(ones), (1,))]
                add_idx = zeros[torch.randint(len(zeros), (1,))]
                new_sol = solution.clone()
                new_sol[remove_idx] = 0
                new_sol[add_idx] = 1
                if self.is_feasible(new_sol):
                    neighbors.append((new_sol, [remove_idx.item(), add_idx.item()]))
        
        return neighbors
    
    def is_tabu(self, move_indices: List[int]) -> bool:
        """Check if any move in the combination is tabu"""
        return any(idx in self.tabu_list for idx in move_indices)
    
    def update_tabu_list(self, move_indices: List[int]):
        """Add moves to tabu list and remove old moves"""
        for idx in move_indices:
            self.tabu_list.append(idx)
        while len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.pop(0)
    
    @torch.no_grad()
    def run(self, n_iterations: int) -> Tuple[float, torch.Tensor]:
        """Run enhanced tabu search"""
        # Generate initial solutions with dynamic size
        n_init = max(5, min(8, self.n // 100))  # Scale with problem size
        solutions = [self.generate_initial_solution() for _ in range(n_init)]
        current_solution = max(solutions, key=self.objective)
        self.best_solution = current_solution.clone()
        self.best_obj = self.objective(current_solution)
        
        no_improvement = 0
        restart_threshold = 25
        min_tenure = 5
        max_tenure = 9
        self.tabu_tenure = min_tenure
        
        for iter_idx in range(n_iterations):
            neighbors = self.get_neighborhood(current_solution)
            if not neighbors:
                continue
            
            # Evaluate neighbors
            best_neighbor = None
            best_neighbor_obj = float('-inf')
            best_move_indices = None
            second_best_neighbor = None
            second_best_obj = float('-inf')
            
            for neighbor, move_indices in neighbors:
                neighbor_obj = self.objective(neighbor)
                
                # Enhanced acceptance criteria
                if not self.is_tabu(move_indices):
                    if neighbor_obj > best_neighbor_obj:
                        second_best_neighbor = best_neighbor
                        second_best_obj = best_neighbor_obj
                        best_neighbor = neighbor
                        best_neighbor_obj = neighbor_obj
                        best_move_indices = move_indices
                    elif neighbor_obj > second_best_obj:
                        second_best_neighbor = neighbor
                        second_best_obj = neighbor_obj
                elif neighbor_obj > self.best_obj:  # Aspiration criterion
                    best_neighbor = neighbor
                    best_neighbor_obj = neighbor_obj
                    best_move_indices = move_indices
            
            if best_neighbor is None:
                no_improvement += 1
                # Try second best if best is None
                if second_best_neighbor is not None:
                    best_neighbor = second_best_neighbor
                    best_neighbor_obj = second_best_obj
                else:
                    continue
            
            # Update current solution
            current_solution = best_neighbor
            
            # Update best solution if improved with dynamic tabu tenure
            if best_neighbor_obj > self.best_obj:
                self.best_solution = best_neighbor.clone()
                self.best_obj = best_neighbor_obj
                no_improvement = 0
                # Reduce tabu tenure when improving
                self.tabu_tenure = max(min_tenure, self.tabu_tenure - 1)
            else:
                no_improvement += 1
                # Increase tabu tenure when not improving
                self.tabu_tenure = min(max_tenure, self.tabu_tenure + 1)
            
            # Enhanced restart strategy with dynamic threshold
            if no_improvement > restart_threshold:
                new_solution = self.generate_initial_solution()
                accept_threshold = 0.85 + 0.05 * (iter_idx / n_iterations)  # Increase acceptance threshold over time
                if self.objective(new_solution) > self.objective(current_solution) * accept_threshold:
                    current_solution = new_solution
                    self.tabu_list.clear()
                no_improvement = 0
            
            # Update tabu list
            self.update_tabu_list(best_move_indices)
        
        return self.best_obj, self.best_solution
