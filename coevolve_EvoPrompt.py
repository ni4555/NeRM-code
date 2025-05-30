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

        # Load initial description from yaml
        self.yaml_path = Path(root_dir) / "cfg" / "problem" / f"{self.problem_name}.yaml"
        with open(self.yaml_path, 'r') as f:
            self.problem_config = yaml.safe_load(f)
            self.initial_desc = self.problem_config['description']

        # Evolution parameters
        self.population_size = 10  # 每个prompt生成的代码个体数量
        self.num_generations = 25
        self.elite_size = 2
        self.mutation_rate = 0.5
        self.num_diff_desc = 10  # 默认记录10个不同适应度的description

        # Initialize iteration counter and unique descriptions tracker
        self.iteration = 0
        self.unique_desc_results = {}  # 用于追踪不同适应度的description
        self.description_population = []  # 存储所有不同prompt及其最佳代码个体

        self.population: List[dict] = []  # 改为存储完整的个体信息

        # Load template for prompt generation
        self.template = """Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: {prompt1}
Prompt 2: {prompt2}
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: {prompt3}

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: {prompt1}
Prompt 2: {prompt2}
Different parts:
{parts}

2. Randomly mutate the different parts:
{mutations}

3. Combine the different parts with Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: {prompt3}

Final Prompt: <prompt>{final_prompt}</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: {prompt1}
Prompt 2: {prompt2}
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: {prompt3}

Important: The new prompt should:
1. Include specific technical details about the optimization problem
2. Emphasize different aspects of the solution strategy
3. Use domain-specific terminology
4. Provide clear guidance on algorithm design
5. Focus on different optimization techniques or approaches

1. """

        # Default prompts for initial generations
        self.default_prompts = [
            "Given a problem description, generate a solution that minimizes the objective value.",
            "Create an algorithm to solve the optimization problem with the lowest possible objective value.",
            "Design an efficient solution strategy that finds the optimal solution with minimal objective value.",
            "Develop a solution approach that optimizes the objective function while considering problem constraints.",
            "Implement a solution method that efficiently explores the solution space to find optimal results."
        ]

    def extract_prompt_from_response(self, response: str) -> str:
        """Extract the prompt from the response text that's bracketed with <prompt> and </prompt>"""
        try:
            # Find the last occurrence of <prompt> and </prompt>
            last_start = response.rfind("<prompt>")
            if last_start == -1:
                return response.strip()

            # Add length of <prompt> to get to the content start
            content_start = last_start + 8

            # Find the next </prompt> after the last <prompt>
            content_end = response.find("</prompt>", content_start)
            if content_end == -1:
                return response.strip()

            # Extract and return the content between the tags
            return response[content_start:content_end].strip()
        except Exception as e:
            print(e)
            return response.strip()

    def generate_variations(self, base_desc: str) -> List[str]:
        """Generate variations of a description using template-based LLM"""
        # Create a simple template for generating variations
        template_prompt = f"""Given this problem description:
{base_desc}

Generate a variation of this description that maintains the same goal but uses different wording or structure.
The variation should be focused on solving the optimization problem effectively.

Return only the variation, bracketed with <prompt> and </prompt>."""

        variations = []
        try:
            messages = [{"role": "user", "content": template_prompt}]
            responses = self.client.multi_chat_completion(
                [messages],
                1,  # Only need one variation at a time
                temperature=0.8
            )
            variations.append(self.extract_prompt_from_response(responses[0]))
        except Exception as e:
            logging.error(f"Error generating variation: {e}")
            variations.append(base_desc)

        return variations

    def crossover(self, desc1: str, desc2: str) -> str:
        """Perform crossover between two descriptions using template-based LLM"""
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

        # Get a third prompt from population with good fitness
        prompt3 = None
        for individual in self.population:
            if individual["description"] not in [desc1, desc2]:
                prompt3 = individual["description"]
                break
        if not prompt3:
            # Use a default prompt if no suitable one found in population
            prompt3 = self.default_prompts[2]

        # Create a simpler template for combining prompts
        template_prompt = f"""Given these two problem descriptions:
Description 1: {desc1}
Description 2: {desc2}

Generate a new problem description that combines the key elements from both descriptions while maintaining the optimization goal.
The new description should be clear, concise, and focused on solving the optimization problem effectively.

Return only the new description, bracketed with <prompt> and </prompt>."""

        try:
            messages = [{"role": "user", "content": template_prompt}]
            responses = self.client.multi_chat_completion(
                [messages],
                1,  # Only need one combined description
                temperature=0.8
            )
            return self.extract_prompt_from_response(responses[0])
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

Return only the implementation code."""

        try:
            print(prompt, "\n\n")
            messages = [{"role": "user", "content": prompt}]
            responses = self.client.multi_chat_completion(
                [messages],
                num_samples,
                temperature=1.5
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
        """Run code in subprocess and return process with improved error handling"""
        # Create a unique directory for this code evaluation
        code_dir = gen_dir / f"code_{individual['response_id']}"
        code_dir.mkdir(exist_ok=True)
        
        # Save code to unique directory for record keeping
        code_path = code_dir / "gpt.py"
        stdout_path = code_dir / "stdout.txt"

        # Update paths in individual
        individual["code_path"] = str(code_path)
        individual["stdout_filepath"] = str(stdout_path)

        # Save code to unique directory for record keeping
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(individual["code"])

        # Save code to problem directory for evaluation
        problem_gpt_path = Path(self.root_dir) / "problems" / self.problem_name / "gpt.py"
        with open(problem_gpt_path, 'w', encoding='utf-8') as f:
            f.write(individual["code"])

        # Create stdout file
        with open(stdout_path, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                ['python', '-u', f"{self.root_dir}/problems/{self.problem_name}/eval.py", f'{self.problem_size}',
                 str(self.root_dir), "train"],  # Use root_dir instead of code_dir
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
        all_individuals = []
        
        # Generate code samples
        logging.info(f"Generating {self.population_size} code samples...")
        codes = self.generate_llm(description, num_samples=self.population_size)
        if not codes:
            return best_fitness, best_code, all_individuals

        # Convert codes to individuals for evaluation
        individuals = []
        for response_id, code in enumerate(codes):
            if not code:
                logging.error(f"Invalid code for response {response_id}")
                continue

            individual = {
                "code": code,
                "response_id": response_id,
                "description": description,
                "exec_success": False,
                "obj": float('inf')
            }
            individuals.append(individual)

        # Evaluate all individuals
        evaluated_individuals = self.evaluate_population(individuals, gen_dir)

        # Find best result for this description
        successful_individuals = [ind for ind in evaluated_individuals if ind.get("exec_success", False)]
        if successful_individuals:
            best_individual = min(successful_individuals, key=lambda x: x.get("obj", float('inf')))
            best_fitness = best_individual["obj"]
            best_code = best_individual["code"]

        return best_fitness, best_code, evaluated_individuals

    def evaluate_population(self, population: List[dict], gen_dir: Path) -> List[dict]:
        """Evaluate a population of individuals with improved error handling and logging"""
        # Process in batches to manage resources
        max_concurrent = min(4, len(population))
        results = []
        
        # Process in batches
        for i in range(0, len(population), max_concurrent):
            batch = population[i:i + max_concurrent]
            running_processes = []
            
            # Launch batch
            for individual in batch:
                try:
                    process_tuple = self._run_code(individual, gen_dir)
                    running_processes.append((process_tuple, individual))
                    logging.info(f"Launched evaluation for code {individual['response_id']}")
                except Exception as e:
                    logging.error(f"Launch failed for code {individual['response_id']}: {e}")
                    individual["exec_success"] = False
                    individual["obj"] = float('inf')
                    individual["error"] = str(e)
                    results.append(individual)
            
            # Wait for batch completion
            for process_tuple, individual in running_processes:
                if process_tuple is None:
                    continue
                
                process, stdout_path, _ = process_tuple
                try:
                    process.communicate(timeout=self.cfg.timeout)
                    
                    with open(stdout_path, 'r', encoding='utf-8') as f:
                        stdout_str = f.read()
                    
                    traceback_msg = filter_traceback(stdout_str)

                    if traceback_msg == '':
                        try:
                            # Get the last non-empty line that contains a number
                            output_lines = [line.strip() for line in stdout_str.split('\n') if line.strip()]
                            for line in reversed(output_lines):
                                try:
                                    fitness = float(line)
                                    individual["exec_success"] = True
                                    individual["obj"] = fitness if self.cfg.problem.obj_type == "min" else -fitness
                                    logging.info(f"Code {individual['response_id']} success: {individual['obj']}")
                                    break
                                except ValueError:
                                    continue
                            else:
                                raise ValueError("No valid fitness value found in output")
                        except (ValueError, IndexError) as e:
                            logging.error(f"Error parsing fitness for code {individual['response_id']}: {e}")
                            individual["exec_success"] = False
                            individual["obj"] = float('inf')
                            individual["error"] = str(e)
                    else:
                        individual["exec_success"] = False
                        individual["obj"] = float('inf')
                        individual["error"] = traceback_msg
                        logging.error(f"Code {individual['response_id']} failed with error: {traceback_msg}")
                    
                    results.append(individual)
                        
                except subprocess.TimeoutExpired:
                    logging.error(f"Code {individual['response_id']} evaluation timed out")
                    process.kill()
                    process.communicate()
                    individual["exec_success"] = False
                    individual["obj"] = float('inf')
                    individual["error"] = "Timeout"
                    results.append(individual)
                except Exception as e:
                    logging.error(f"Error processing results for code {individual['response_id']}: {e}")
                    if process.poll() is None:
                        process.kill()
                    individual["exec_success"] = False
                    individual["obj"] = float('inf')
                    individual["error"] = str(e)
                    results.append(individual)

        return results

    def is_description_unique(self, description: str) -> bool:
        """
        检查description文本是否是唯一的
        """
        for individual in self.description_population:
            if individual["description"] == description:
                return False
        return True

    def evolve(self) -> Tuple[str, float, str]:
        """Main evolution loop"""
        logging.info("Starting evolution...")
        self.description_population = []  # 重置description种群
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

            # Generate next description if not done
            if gen < self.num_generations and len(self.unique_desc_results) < self.num_diff_desc:
                print("len desc pop: ", len(self.description_population))
                print(self.description_population)
                if len(self.description_population) >= 3:  # When we have enough samples, use template-based evolution
                    # Sort population by fitness
                    sorted_population = sorted(self.description_population, key=lambda x: x["obj"])
                    # Get top 3 prompts
                    prompt1 = sorted_population[0]["description"]
                    prompt2 = sorted_population[1]["description"]
                    prompt3 = sorted_population[2]["description"]

                    # Create example parts and mutations with more technical focus
                    parts = '"Given a problem description" vs "Create an algorithm"\n"minimizes the objective value" vs "lowest possible objective value"'
                    mutations = '"Given a problem description" -> "For the given optimization problem"\n"minimizes the objective value" -> "achieves the optimal solution"'
                    final_prompt = "For the given optimization problem, develop a solution strategy that achieves the optimal solution with minimal objective value."

                    # Format the template
                    template_prompt = self.template.format(
                        prompt1=prompt1,
                        prompt2=prompt2,
                        prompt3=prompt3,
                        parts=parts,
                        mutations=mutations,
                        final_prompt=final_prompt
                    )

                    try:
                        messages = [{"role": "user", "content": template_prompt}]
                        responses = self.client.multi_chat_completion(
                            [messages],
                            1,
                            temperature=0.8
                        )

                        print("prompt 1: ", prompt1)
                        print("prompt 2: ", prompt2)
                        print("prompt 3: ", prompt3)
                        print("responses: ", responses)
                        current_desc = self.extract_prompt_from_response(responses[0])
                        print("current desc: ", current_desc)
                    except Exception as e:
                        logging.error(f"Error in template-based evolution: {e}")
                        # Fallback to crossover if template fails
                        parent1, parent2 = random.sample(self.description_population[:max(2, len(self.description_population))], 2)
                        current_desc = self.crossover(parent1["description"], parent2["description"])
                elif len(self.description_population) >= 2:  # When we have 2 samples, use crossover
                    parent1, parent2 = random.sample(self.description_population, 2)
                    current_desc = self.crossover(parent1["description"], parent2["description"])
                else:  # When we have less than 2 samples, use variations
                    variations = self.generate_variations(current_desc)
                    if variations:
                        current_desc = variations[0]

            # Update YAML with new description before evaluation
            self.problem_config['description'] = current_desc
            temp_yaml_path = self.output_dir / "temp_config.yaml"
            with open(temp_yaml_path, 'w') as f:
                yaml.dump(self.problem_config, f)

            # Evaluate current description
            fitness, code, evaluated_individuals = self.evaluate_description(current_desc, gen_dir)

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

            # Update description population and track results
            if self.is_description_unique(current_desc):  # 只检查描述文本是否唯一
                self.description_population.append(current_individual)
                self.description_population.sort(key=lambda x: x["obj"])
                # # Only keep the best population_size descriptions
                # if len(self.description_population) > self.population_size:
                #     self.description_population = self.description_population[:self.population_size]

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
