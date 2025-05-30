import torch
import numpy as np
from typing import Tuple

class VariableNeighborhoodSearch:
    def __init__(self,
                 prize,    # shape [n,]
                 weight,   # shape [m, n]
                 heuristic,
                 n_solutions=10,  # number of initial solutions 
                 device='cpu'
                 ):
        self.n, self.m = weight.shape
        self.prize = prize
        self.weight = weight
        self.heuristic = heuristic
        self.n_solutions = n_solutions
        self.device = device
        
        # 预计算效率比
        self.efficiency = prize / weight.sum(dim=1)
        
        self.best_solution = None
        self.best_objective = 0
        
    def is_feasible(self, solution: torch.Tensor) -> bool:
        """Check if solution satisfies all constraints"""
        if len(solution) == 0:
            return True
        total_weight = self.weight[solution].sum(dim=0)
        return torch.all(total_weight <= 1.0)

    def calculate_objective(self, solution: torch.Tensor) -> float:
        """Calculate objective value for a solution"""
        if len(solution) == 0:
            return 0.0
        return self.prize[solution].sum().item()

    def generate_initial_solution(self) -> torch.Tensor:
        """Generate initial solution using greedy approach with efficiency ratio"""
        solution = []
        current_weight = torch.zeros(self.m, device=self.device)
        
        # 按效率比排序
        values, indices = self.efficiency.sort(descending=True)
        
        # 贪心选择前50%的高效率物品尝试放入
        top_k = int(self.n * 0.5)
        for idx in indices[:top_k]:
            if torch.all(current_weight + self.weight[idx] <= 1.0):
                solution.append(idx.item())
                current_weight += self.weight[idx]
                
        return torch.tensor(solution, device=self.device)

    def swap_neighborhood(self, solution: torch.Tensor) -> torch.Tensor:
        """Swap one item in solution with one item not in solution"""
        if len(solution) == 0:
            return solution
            
        best_neighbor = solution.clone()
        best_value = self.calculate_objective(solution)
        
        # 只考虑效率比较高的物品进行交换
        top_k = int(self.n * 0.3)  # 只考虑前30%的高效率物品
        values, candidates = self.efficiency.sort(descending=True)
        candidates = candidates[:top_k]
        
        for i in range(len(solution)):
            for j in candidates:
                if j not in solution:
                    new_solution = solution.clone()
                    new_solution[i] = j
                    
                    if self.is_feasible(new_solution):
                        new_value = self.calculate_objective(new_solution)
                        if new_value > best_value:
                            best_neighbor = new_solution.clone()
                            best_value = new_value
                            
        return best_neighbor

    def add_remove_neighborhood(self, solution: torch.Tensor) -> torch.Tensor:
        """Try to add one item or remove one item"""
        best_neighbor = solution.clone()
        best_value = self.calculate_objective(solution)
        
        # 只考虑效率比较高的物品进行添加
        top_k = int(self.n * 0.3)
        values, candidates = self.efficiency.sort(descending=True)
        candidates = candidates[:top_k]
        
        # Try adding one item
        for i in candidates:
            if i not in solution:
                new_solution = torch.cat([solution, torch.tensor([i], device=self.device)])
                if self.is_feasible(new_solution):
                    new_value = self.calculate_objective(new_solution)
                    if new_value > best_value:
                        best_neighbor = new_solution.clone()
                        best_value = new_value
        
        # Try removing one item
        if len(solution) > 0:
            for i in range(len(solution)):
                new_solution = torch.cat([solution[:i], solution[i+1:]])
                new_value = self.calculate_objective(new_solution)
                if new_value > best_value:
                    best_neighbor = new_solution.clone()
                    best_value = new_value
                        
        return best_neighbor

    @torch.no_grad()
    def run(self, n_iterations: int) -> Tuple[float, torch.Tensor]:
        """Run the VNS algorithm"""
        # Generate initial solutions
        solutions = [self.generate_initial_solution() for _ in range(self.n_solutions)]
        self.best_solution = solutions[0]
        self.best_objective = self.calculate_objective(self.best_solution)
        
        neighborhoods = [
            self.swap_neighborhood,
            self.add_remove_neighborhood
        ]
        
        no_improve_count = 0
        max_no_improve = 20  # 如果连续20次没有改善就提前停止
        
        for iter in range(n_iterations):
            old_best = self.best_objective
            
            for solution in solutions:
                # Apply each neighborhood structure
                current_solution = solution.clone()
                
                # 随机选择邻域结构
                k = np.random.randint(0, len(neighborhoods))
                neighbor = neighborhoods[k](current_solution)
                current_value = self.calculate_objective(neighbor)
                
                # Update best solution if necessary
                if current_value > self.best_objective:
                    self.best_solution = neighbor.clone()
                    self.best_objective = current_value
            
            # 检查是否有改善
            if self.best_objective <= old_best:
                no_improve_count += 1
            else:
                no_improve_count = 0
                
            # 如果长时间没有改善，提前终止
            if no_improve_count >= max_no_improve:
                break
        
        return self.best_objective, self.best_solution

if __name__ == '__main__':
    # Test code
    from gen_inst import gen_instance
    n, m = 50, 5
    prize, weight = gen_instance(n, m)
    heu = np.random.rand(n)
    vns = VariableNeighborhoodSearch(
        torch.from_numpy(prize), 
        torch.from_numpy(weight),
        torch.from_numpy(heu),
        n_solutions=10
    )
    obj, _ = vns.run(100)
    print(f"Best objective: {obj}")
