# -*- coding: gbk -*-
import json
import os
import yaml
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import subprocess
import shutil
from utils.utils import extract_code_from_generator, filter_traceback, block_until_running


class DescriptionCoevolver:
    def __init__(self, cfg, root_dir, client):
        self.cfg = cfg
        self.problem_size = self.cfg.problem.problem_size
        self.root_dir = root_dir
        self.client = client
        self.problem_name = cfg.problem.problem_name

        # Load seed function for code extraction reference
        seed_func_path = f"{root_dir}/prompts/{self.problem_name}/seed_func.txt"
        with open(seed_func_path, 'r') as f:
            self.seed_func = f.read().strip()

        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"coevolve")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create initial gpt.py file if it doesn't exist
        gpt_file = Path(root_dir) / "problems" / self.problem_name / "gpt.py"
        if not gpt_file.exists():
            gpt_file.parent.mkdir(parents=True, exist_ok=True)
            with open(gpt_file, 'w') as f:
                # Write initial empty function or seed code
                f.write(extract_code_from_generator(self.seed_func).replace("v1", "v2"))

        # Load initial description
        self.yaml_path = Path(root_dir) / "cfg" / "problem" / f"{self.problem_name}.yaml"
        with open(self.yaml_path, 'r') as f:
            self.problem_config = yaml.safe_load(f)
            self.initial_desc = self.problem_config['description']

        # Evolution parameters
        self.population_size = 12
        self.num_generations = 10
        self.elite_size = 2
        self.mutation_rate = 0.5
        self.num_diff_desc = 10  # 默认记录5个不同适应度的description

        # Initialize iteration counter and unique descriptions tracker
        self.iteration = 0
        self.unique_desc_results = {}  # 用于追踪不同适应度的description

        self.population: List[dict] = []  # 改为存储完整的个体信息

    def generate_variations(self, base_desc: str) -> List[str]:
        """Generate variations of a description using LLM"""
        prompt = f"""Given this problem description:

{base_desc}

Generate a variation of this description that might lead to better solution strategies. 
The description should:
1. Be clear and specific about the solution approach
2. Focus on key algorithmic aspects and optimization techniques
3. Maintain similar length but vary the emphasis or perspective

Return only the new description text, without any additional explanation."""

        variations = []
        try:
            messages = [{"role": "user", "content": prompt}]
            responses = self.client.multi_chat_completion(
                [messages],
                1,  # Only need one variation at a time
                temperature=0.8
            )
            variations.append(responses[0])
        except Exception as e:
            logging.error(f"Error generating variation: {e}")
            variations.append(base_desc)

        return variations

    def crossover(self, desc1: str, desc2: str) -> str:
        """Perform crossover between two descriptions using LLM"""
        # Get the corresponding codes and fitness values
        code1 = ""
        code2 = ""
        fitness1 = float('inf')
        fitness2 = float('inf')

        # Find the corresponding individuals for the descriptions
        for individual in self.population:
            if individual["description"] == desc1:
                code1 = individual["code"]
                fitness1 = individual["obj"]
            if individual["description"] == desc2:
                code2 = individual["code"]
                fitness2 = individual["obj"]

        # Prepare implementation strings separately to avoid f-string issues
        impl1 = f'Implementation 1:\n{code1}' if code1 else ''
        impl2 = f'Implementation 2:\n{code2}' if code2 else ''

        prompt = f"""Combine these two problem descriptions into a new one that captures the best elements of both:

Description 1 (fitness: {fitness1}):
{desc1}

{impl1}

Description 2 (fitness: {fitness2}):
{desc2}

{impl2}

Create a single cohesive description that:
1. Combines the most effective aspects of both descriptions
2. Maintains clarity and specificity
3. Emphasizes elements that could lead to solutions better than {min(fitness1, fitness2)}

Return only the new combined description, without any additional explanation."""

        try:
            messages = [{"role": "user", "content": prompt}]
            responses = self.client.multi_chat_completion(
                [messages],
                1,  # Only need one combined description
                temperature=0.8
            )
            return responses[0]
        except Exception as e:
            logging.error(f"Error in crossover: {e}")
            return random.choice([desc1, desc2])

    def generate_llm(self, description: str, num_samples: int = 10) -> List[str]:
        """Generate multiple code samples using LLM"""
        codes = []

        # Load function signature and description
        with open(f"{self.root_dir}/prompts/{self.problem_name}/func_signature.txt", 'r') as f:
            func_signature = f.read().strip()

        # Replace version in function signature
        func_signature = func_signature.format(version=2)  # Always use v2 for generated code

        with open(f"{self.root_dir}/prompts/{self.problem_name}/func_desc.txt", 'r') as f:
            func_desc = f.read().strip()

        prompt = f"""Based on this problem description:
{description}

And following this function description:
{func_desc}

Implement the solution using this function signature:
{func_signature}

Example implementation you can refer:
{self.seed_func.replace("v1", "v2")}

Return only the implementation code."""

        try:
            messages = [{"role": "user", "content": prompt}]
            responses = self.client.multi_chat_completion(
                [messages],
                num_samples,
                temperature=0.8
            )

            # Extract valid code from each response
            for response in responses:
                try:
                    # Extract code and replace version if needed
                    code = extract_code_from_generator(response)
                    if code:
                        code = code.replace("v1", "v2")  # Version replacement if needed
                        codes.append(code)  # No need to replace v1 with v2 here since signature already has v2
                except Exception as e:
                    logging.error(f"Error extracting code: {e}")
                    continue

        except Exception as e:
            logging.error(f"Error in LLM generation: {e}")

        return codes

    def _run_code(self, individual: dict, gen_dir: Path) -> Tuple[subprocess.Popen, str, str]:
        """Run code in subprocess and return process"""
        # Save code to generation directory
        code_path = gen_dir / f"code_{individual['response_id']}.py"
        stdout_path = gen_dir / f"stdout_{individual['response_id']}.txt"

        # Update paths in individual
        individual["code_path"] = str(code_path)
        individual["stdout_filepath"] = str(stdout_path)

        # Save code to file
        with open(code_path, 'w') as f:
            f.write(individual["code"])

        # Save code to main gpt.py file for evaluation
        with open(f"{self.root_dir}/problems/{self.problem_name}/gpt.py", 'w') as f:
            f.write(individual["code"])

        # Create stdout file
        with open(stdout_path, 'w') as f:
            process = subprocess.Popen(
                ['python', '-u', f"{self.root_dir}/problems/{self.problem_name}/eval.py", f'{self.problem_size}',
                 self.root_dir, "train"],
                stdout=f,
                stderr=f
            )

        # Wait for process to start
        block_until_running(stdout_path, log_status=True, iter_num=self.iteration,
                            response_id=individual["response_id"])
        return process, stdout_path, code_path

    def evaluate_description(self, description: str, gen_dir: Path) -> Tuple[float, str, List[dict]]:
        """Evaluate a description by generating and testing multiple solutions"""
        # Update YAML with new description
        self.problem_config['description'] = description
        temp_yaml_path = self.output_dir / "temp_config.yaml"
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(self.problem_config, f)

        best_fitness = float('inf')
        best_code = ""

        # Generate population_size solutions for this description
        codes = self.generate_llm(description, num_samples=self.population_size)

        # Convert codes to individuals for evaluation
        individuals = []
        for response_id, code in enumerate(codes):
            if not code:
                logging.error(f"Invalid code for response {response_id}")
                continue

            individual = {
                "code": code,
                "response_id": response_id,
                "description": description
            }
            individuals.append(individual)

        # Evaluate all individuals
        evaluated_individuals = self.evaluate_population(individuals, gen_dir)

        # Find best result for this description
        for individual in evaluated_individuals:
            if individual.get("exec_success", False):
                fitness = individual.get("obj", float('inf'))
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_code = individual["code"]

        return best_fitness, best_code, evaluated_individuals

    def evaluate_population(self, population: List[dict], gen_dir: Path) -> List[dict]:
        """Evaluate a population of individuals"""
        # Start all evaluations
        running_processes = []
        for individual in population:
            try:
                process, stdout_file, code_file = self._run_code(individual, gen_dir)
                running_processes.append((process, individual))
            except Exception as e:
                logging.error(f"Error starting evaluation for code {individual['response_id']}: {e}")
                individual["exec_success"] = False
                individual["obj"] = float('inf')
                individual["error"] = str(e)

        # Wait for all processes and collect results
        for process, individual in running_processes:
            try:
                process.communicate(timeout=self.cfg.timeout)

                with open(individual["stdout_filepath"], 'r') as f:
                    stdout_str = f.read()

                traceback_msg = filter_traceback(stdout_str)

                if traceback_msg == '':
                    try:
                        fitness = float(stdout_str.split('\n')[-2]) if self.cfg.problem.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                        individual["exec_success"] = True
                        individual["obj"] = fitness
                    except (ValueError, IndexError) as e:
                        logging.error(f"Error parsing fitness for code {individual['response_id']}: {e}")
                        individual["exec_success"] = False
                        individual["obj"] = float('inf')
                        individual["error"] = str(e)
                else:
                    individual["exec_success"] = False
                    individual["obj"] = float('inf')
                    individual["error"] = traceback_msg

            except subprocess.TimeoutExpired:
                logging.error(f"Code {individual['response_id']} evaluation timed out")
                process.kill()
                process.communicate()
                individual["exec_success"] = False
                individual["obj"] = float('inf')
                individual["error"] = "Timeout"
            except Exception as e:
                logging.error(f"Error processing results for code {individual['response_id']}: {e}")
                if process.poll() is None:
                    process.kill()
                individual["exec_success"] = False
                individual["obj"] = float('inf')
                individual["error"] = str(e)

        return population

    def is_fitness_unique(self, fitness: float, tolerance: float = 1e-6) -> bool:
        """
        检查适应度是否是唯一的（考虑浮点数误差）
        """
        for existing_fitness in self.unique_desc_results.keys():
            if abs(existing_fitness - fitness) < tolerance:
                return False
        return True

    def add_unique_description(self, description: str, fitness: float, code: str) -> bool:
        """
        添加新的唯一description到记录中
        返回是否成功添加（是否是唯一的）
        """
        if self.is_fitness_unique(fitness):
            self.unique_desc_results[fitness] = {
                "description": description,
                "code": code
            }
            return True
        return False

    def evolve(self) -> Tuple[str, float, str]:
        """Main evolution loop"""
        logging.info("Starting evolution...")
        self.population = []
        current_desc = self.initial_desc
        best_desc = None
        best_fitness = float('inf')
        best_code = None

        # Track global evolution process
        global_results = {
            'generations': []
        }

        gen = 0
        while gen < self.num_generations and len(self.unique_desc_results) < self.num_diff_desc:
            logging.info(f"Generation {gen + 1}/{self.num_generations}")
            self.iteration = gen

            # Create generation directory
            gen_dir = self.output_dir / f"generation_{gen}"
            gen_dir.mkdir(exist_ok=True)

            # Evaluate current description
            fitness, code, evaluated_individuals = self.evaluate_description(current_desc, gen_dir)

            # 只有当适应度是唯一的时才记录这个generation
            if self.add_unique_description(current_desc, fitness, code):
                # Create individual for current description
                current_individual = {
                    "description": current_desc,
                    "obj": fitness,
                    "code": code,
                    "exec_success": True if fitness != float('inf') else False,
                    "error": None
                }

                # Save generation results with all evaluated individuals
                self.save_generation_results(gen, gen_dir, current_desc, evaluated_individuals)

                # Update population and track results
                self.population.append(current_individual)
                self.population.sort(key=lambda x: x["obj"])
                if len(self.population) > self.population_size:
                    self.population = self.population[:self.population_size]

                # Update best solution
                if fitness < best_fitness:
                    best_desc = current_desc
                    best_fitness = fitness
                    best_code = code

                # Track generation in global results
                global_results['generations'].append({
                    'generation': gen,
                    'description': current_desc,
                    'best_fitness': float(fitness),
                    'best_code': code
                })

                gen += 1
            else:
                logging.info(f"Skipping duplicate fitness value: {fitness}")

            # Generate next description if not done
            if gen < self.num_generations and len(self.unique_desc_results) < self.num_diff_desc:
                if len(self.population) >= 2:
                    parent1, parent2 = random.sample(self.population[:max(2, len(self.population))], 2)
                    current_desc = self.crossover(parent1["description"], parent2["description"])

                    if random.random() < self.mutation_rate:
                        variations = self.generate_variations(current_desc)
                        if variations:
                            current_desc = variations[0]
                else:
                    variations = self.generate_variations(current_desc)
                    if variations:
                        current_desc = variations[0]

        # Sort generations by best_fitness before saving
        global_results['generations'].sort(key=lambda x: x['best_fitness'])

        # Add unique descriptions summary to global results
        global_results['unique_descriptions'] = [
            {
                'fitness': float(fitness),
                'description': info['description'],
                'code': info['code']
            }
            for fitness, info in sorted(self.unique_desc_results.items())
        ]

        # Save global results
        with open(self.output_dir / 'global_coevolve.json', 'w') as f:
            json.dump(global_results, f, indent=2)

        # Get the best description from sorted global results
        best_generation = global_results['generations'][0]  # First one after sorting is the best
        best_desc = best_generation['description']
        best_fitness = best_generation['best_fitness']
        best_code = best_generation['best_code']

        # Update original YAML with best description
        self.problem_config['description'] = best_desc
        with open(self.yaml_path, 'w') as f:
            yaml.dump(self.problem_config, f)

        # Update seed_func.txt with best code
        seed_func_path = f"{self.root_dir}/prompts/{self.problem_name}/seed_func.txt"
        best_code_v1 = best_code.replace("heuristics_v2", "heuristics_v1")  # Replace function name
        with open(seed_func_path, 'w') as f:
            f.write(best_code_v1)

        logging.info(f"Evolution completed with {len(self.unique_desc_results)} unique descriptions")
        logging.info(f"Updated seed function with best code (fitness: {best_fitness})")
        return best_desc, best_fitness, best_code

    def save_generation_results(self, generation: int, gen_dir: Path, description: str,
                                evaluated_individuals: List[dict]):
        """Save results for current generation"""
        gen_dir.mkdir(exist_ok=True)

        # Separate successful and failed solutions
        successful_solutions = []
        failed_solutions = []

        for individual in evaluated_individuals:
            solution = {
                'code': individual["code"],
                'fitness': float(individual.get("obj", float('inf'))),
                'execution_success': individual.get("exec_success", False),
                'error': individual.get("error", None),
                'stdout_file': individual.get("stdout_filepath", None),
                'code_file': individual.get("code_path", None)
            }

            if individual.get("exec_success", False):
                successful_solutions.append(solution)
            else:
                failed_solutions.append(solution)

        # Sort successful solutions by fitness
        successful_solutions.sort(key=lambda x: x['fitness'])

        # Save successful solutions to results.json
        results = {
            'generation': generation,
            'description': description,
            'solutions': successful_solutions
        }

        with open(gen_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Save failed solutions to results_err.json
        error_results = {
            'generation': generation,
            'description': description,
            'failed_solutions': failed_solutions
        }

        with open(gen_dir / 'results_err.json', 'w') as f:
            json.dump(error_results, f, indent=2)

        # Find best solution in this generation (only from successful solutions)
        successful_individuals = [ind for ind in evaluated_individuals if ind.get("exec_success", False)]
        if successful_individuals:
            best_individual = min(successful_individuals, key=lambda x: x.get("obj", float('inf')))
        else:
            best_individual = evaluated_individuals[0]  # If no successful solutions, use the first one

        # Save best solution separately
        with open(gen_dir / 'best_solution.txt', 'w') as f:
            f.write(f"Fitness: {best_individual.get('obj', float('inf'))}\n\n")
            f.write(f"Description:\n{description}\n\n")
            f.write(f"Code:\n{best_individual['code']}")


def main(cfg, root_dir, client):
    coevolver = DescriptionCoevolver(cfg, root_dir, client)
    best_desc, best_fitness, best_code = coevolver.evolve()
    logging.info(f"Best fitness achieved: {best_fitness}")
    return best_desc, best_fitness, best_code
