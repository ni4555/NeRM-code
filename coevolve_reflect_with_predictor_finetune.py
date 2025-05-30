import os
import signal
import json
import yaml
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import subprocess
from utils.utils import *


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

        # Evolution parameters with optimized defaults
        self.population_size = 4  # Reduced for faster iterations
        self.eval_batch_size = 4  # Match with population size
        self.num_samples = 12  # Generate more samples initially
        self.num_generations = 11  # Reduced for efficiency
        self.elite_size = 2
        self.mutation_rate = 0.5  # Reduced for more stability
        
        # Initialize iteration counter and unique descriptions tracker
        self.iteration = 0
        self.unique_desc_results = {}  # Track different fitness descriptions
        self.population: List[dict] = []  # Store completeindividual information

        # Add reflection tracking
        self.long_term_reflection_str = ""
        
        # Load reflection prompts
        self.system_generator_prompt = file_to_string(f'{self.root_dir}/prompts/common/system_generator.txt')
        self.system_reflector_prompt = file_to_string(f'{self.root_dir}/prompts/common/system_reflector.txt')
        self.user_reflector_st_prompt = file_to_string(f'{self.root_dir}/prompts/common/coevolve/user_reflector_st.txt')
        self.user_reflector_lt_prompt = file_to_string(f'{self.root_dir}/prompts/common/coevolve/user_reflector_lt.txt')
        self.crossover_prompt = file_to_string(f'{self.root_dir}/prompts/common/coevolve/crossover.txt')

        # Initialize embedding cache
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize performance predictor
        self.performance_predictor = None
        predictor_model_path = f"{root_dir}/predictors/{self.problem_name}/best_performance_predictor.pth"
        if os.path.exists(predictor_model_path):
            from predictors.code_performance_predictor_simple_finetune import CodePerformancePredictor
            self.performance_predictor = CodePerformancePredictor()
            self.performance_predictor.load_model(predictor_model_path)
            logging.info("Loaded code performance predictor")
        else:
            logging.warning("Performance predictor model not found. Will evaluate all generated code.")

        # Performance tracking
        self._predictor_success_count = 0
        self._predictor_total_count = 0
        
        # Add code history pool
        self.code_history_pool = []
        self.max_history_pool_size = 30
        
        # Simplified diversity parameters
        self.diversity_threshold = 0.92
        self.min_diversity_samples = 4
        self.max_retries = 3
        
        # Add base temperature
        self.base_temperature = getattr(client, 'temperature', 1)

        # Add timing loggers
        self.api_log_path = "api_invoke_log.log"
        self.eval_batch_log_path = "eval_time_log_batch.log"
        self.eval_single_log_path = "eval_time_log_single.log"
        
        # Initialize counters and timers
        self.api_call_times = []
        self.eval_batch_times = []
        self.eval_single_times = []
        
        # Create log files with headers
        for log_path in [self.api_log_path, self.eval_batch_log_path, self.eval_single_log_path]:
            # log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("Total count: 0, Total time: 0.0s\n")

    def get_dynamic_evaluation_ratio(self) -> float:
        """
        Get dynamic ratio of candidates to evaluate based on predictor success rate
        """
        if self._predictor_total_count == 0:
            return 0.5  # Start with evaluating 50%
        
        success_rate = self._predictor_success_count / self._predictor_total_count
        
        # Adjust ratio based on success rate
        if success_rate > 0.8:
            return 0.2  # If predictor is very accurate, evaluate fewer
        elif success_rate > 0.6:
            return 0.3
        elif success_rate > 0.4:
            return 0.4
        else:
            return 0.5  # If predictor is not reliable, evaluate more

    def update_predictor_stats(self, evaluated_individuals: list[dict]) -> None:
        """Update predictor success statistics with improved tracking"""
        for individual in evaluated_individuals:
            if individual.get("exec_success", False):
                # Get prediction score if available
                pred_score = individual.get("prediction_score", 0.0)
                
                # Consider prediction successful if:
                # 1. Individual is better than current best
                # 2. Prediction score was high for good solution
                current_best = min([ind.get("obj", float('inf')) for ind in self.population]) if self.population else float('inf')
                
                if individual["obj"] < current_best or (pred_score > 0.7 and individual["obj"] < float('inf')):
                    self._predictor_success_count += 1
                    
            self._predictor_total_count += 1

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
                temperature=self.base_temperature + 0.3
            )
            variations.append(responses[0])
        except Exception as e:
            logging.error(f"Error generating variation: {e}")
            variations.append(base_desc)

        return variations

    def crossover(self, desc1: str, desc2: str) -> str:
        """Enhanced crossover using reflection insights with timing"""
        import time
        start_time = time.time()
        
        # First generate reflection
        reflection = self.short_term_reflection(desc1, desc2)
        
        system = self.system_generator_prompt
        user = self.crossover_prompt.format(
            desc1=desc1["description"],
            desc2=desc2["description"],
            reflection=reflection
        )
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        return self.client.multi_chat_completion([messages])[0]

    def generate_llm(self, description: str, num_samples: int = 12) -> List[str]:
        """Generate multiple code samples using LLM with timing tracking"""
        import time
        start_time = time.time()
        
        codes = []
        
        # Load necessary prompts
        with open(f"{self.root_dir}/prompts/{self.problem_name}/func_signature.txt", 'r') as f:
            func_signature = f.read().strip().format(version=2)
        with open(f"{self.root_dir}/prompts/{self.problem_name}/func_desc.txt", 'r') as f:
            func_desc = f.read().strip()

        # Simplified diversity prompts
        diversity_prompts = [
            "Focus on speed optimization.",
            "Focus on memory efficiency.",
            "Use different data structures.",
            "Try an alternative algorithm."
        ]

        try:
            base_temperature = self.client.temperature
            remaining_samples = num_samples
            retry_count = 0

            while remaining_samples > 0 and retry_count < self.max_retries:
                batch_size = min(4, remaining_samples)  # Process in smaller batches
                
                # Adjust temperature based on retry count
                temperature = min(1.2, base_temperature + (0.2 * retry_count))
                
                # Create prompts for batch
                messages_batch = []
                for i in range(batch_size):
                    diversity_prompt = diversity_prompts[i % len(diversity_prompts)]
                    system = self.system_generator_prompt
                    user = f"""Based on this problem description:
{description}

Function description:
{func_desc}

Additional focus: {diversity_prompt}

Implement using this signature:
{func_signature}

Return only the implementation code."""

                    messages_batch.append([
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ])

                # Generate responses in batch
                responses = self.client.multi_chat_completion(
                    messages_batch,
                    1,  # One response per message
                    temperature=temperature
                )

                # Process responses
                new_codes = []
                for response in responses:
                    try:
                        code = extract_code_from_generator(response)
                        if code:
                            code = code.replace("v1", "v2")
                            new_codes.append(code)
                    except Exception as e:
                        logging.error(f"Error extracting code: {e}")
                        continue

                # Check diversity for new codes
                for code in new_codes:
                    if len(codes) < self.min_diversity_samples or \
                       self._is_code_diverse(code, codes, self.diversity_threshold):
                        codes.append(code)
                        remaining_samples -= 1
                        if remaining_samples <= 0:
                            break

                retry_count += 1

            logging.info(f"Generated {len(codes)} diverse codes (after {retry_count} attempts)")
            
        except Exception as e:
            logging.error(f"Error in LLM generation: {e}")

        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        return codes[:num_samples]  # Ensure we don't return more than requested

    def _run_code(self, individual: dict, gen_dir: Path) -> Tuple[subprocess.Popen, str, str]:
        """Run code in subprocess and return process with timeout handling"""
        code_path = gen_dir / f"code_{individual['response_id']}.py"
        stdout_path = gen_dir / f"stdout_{individual['response_id']}.txt"

        individual["code_path"] = str(code_path)
        individual["stdout_filepath"] = str(stdout_path)

        # Save code to files
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(individual["code"] + '\n')
        with open(Path(self.root_dir) / "problems" / self.problem_name / "gpt.py", 'w', encoding='utf-8') as f:
            f.write(individual["code"] + '\n')

        # Create stdout file and launch process
        with open(stdout_path, 'w', encoding='utf-8') as f:
            eval_file_path = f'{self.root_dir}/problems/{self.problem_name}/eval.py'
            process = subprocess.Popen(
                ['python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train"],
                stdout=f, 
                stderr=f
            )
        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=individual['response_id'])
        return process, stdout_path, code_path

    def _evaluate_parallel(self, population: list[dict], gen_dir: Path) -> list[dict]:
        """Optimized parallel evaluation with resource management and timing"""
        import time
        batch_start = time.time()
        
        max_concurrent = min(4, len(population))
        timeout = self.cfg.timeout
        results = []
        
        # Process in batches to manage resources
        for i in range(0, len(population), max_concurrent):
            batch = population[i:i + max_concurrent]
            processes = []
            
            # Launch batch
            for individual in batch:
                try:
                    single_start = time.time()
                    process_tuple = self._run_code(individual, gen_dir)
                    processes.append((process_tuple, individual, single_start))
                    logging.info(f"Launched evaluation for code {individual['response_id']}")
                except Exception as e:
                    logging.error(f"Launch failed for code {individual['response_id']}: {e}")
                    individual["exec_success"] = False
                    individual["obj"] = float('inf')
                    individual["error"] = str(e)
                    results.append(individual)
            
            # Wait for batch completion
            for process_tuple, individual, start_time in processes:
                if process_tuple is None:
                    continue
                
                process, stdout_path, _ = process_tuple
                try:
                    process.wait(timeout=timeout)
                    single_duration = time.time() - start_time
                    self._log_timing(single_duration, "eval_single")
                    
                    with open(stdout_path, 'r', encoding='utf-8') as f:
                        stdout_str = f.read()
                    
                    if filter_traceback(stdout_str) == '':
                        try:
                            individual['obj'] = float(stdout_str.split('\n')[-2]) if self.cfg.problem.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                            individual["exec_success"] = True
                            logging.info(f"Code {individual['response_id']} success: {individual['obj']}")
                        except:
                            self._mark_failed(individual, "Invalid output format")
                    else:
                        self._mark_failed(individual, filter_traceback(stdout_str))
                        
                except subprocess.TimeoutExpired:
                    self._mark_failed(individual, "Timeout")
                    self._kill_process(process)
                except Exception as e:
                    self._mark_failed(individual, str(e))
                    self._kill_process(process)
                
                results.append(individual)
        
        batch_duration = time.time() - batch_start
        self._log_timing(batch_duration, "eval_batch")
        return results

    def _mark_failed(self, individual: dict, error: str):
        """Mark individual as failed with logging"""
        individual["exec_success"] = False
        individual["obj"] = float('inf')
        individual["error"] = error
        logging.error(f"Code {individual['response_id']} failed: {error}")

    def _kill_process(self, process: subprocess.Popen):
        """Safely kill a process and its children"""
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            if process.poll() is None:
                process.kill()

    def evaluate_description(self, description: str, gen_dir: Path) -> Tuple[float, str, List[dict]]:
        """Optimized evaluation strategy focusing on generation fairness"""
        self.problem_config['description'] = description
        temp_yaml_path = gen_dir / "temp_config.yaml"
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(self.problem_config, f)

        best_fitness = float('inf')
        best_code = ""
        all_individuals = []
        
        # 1. ï¿½ï¿½ï¿½É´ï¿½ï¿½ï¿½
        logging.info(f"Generating {self.num_samples} code samples...")
        codes = self.generate_llm(description, num_samples=self.num_samples)
        if not codes:
            return best_fitness, best_code, all_individuals
        
        # 2. ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½È¡embeddings
        candidates = []
        embeddings_to_get = []
        code_to_idx = {}
        
        for i, code in enumerate(codes):
            code_hash = hash(code)
            if code_hash in self.embedding_cache:
                candidates.append({
                    "code": code,
                    "response_id": i,
                    "description": description,
                    "embedding": self.embedding_cache[code_hash],
                    "exec_success": False,
                    "obj": float('inf')
                })
            else:
                embeddings_to_get.append(code)
                code_to_idx[code] = i
        
        # ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½È¡ï¿½Âµï¿½embeddings
        if embeddings_to_get:
            try:
                from utils.get_embeddings import batch_embedding_api
                new_embeddings = batch_embedding_api(
                    self.cfg.embedding_llm_model,
                    self.cfg.embedding_end_point,
                    self.cfg.embedding_api_key,
                    embeddings_to_get
                )
                
                for code, embedding in zip(embeddings_to_get, new_embeddings):
                    if embedding is not None:
                        code_hash = hash(code)
                        self.embedding_cache[code_hash] = embedding
                        candidates.append({
                            "code": code,
                            "response_id": code_to_idx[code],
                            "description": description,
                            "embedding": embedding,
                            "exec_success": False,
                            "obj": float('inf')
                        })
            except Exception as e:
                logging.error(f"Batch embedding error: {e}")
        
        if not candidates:
            return best_fitness, best_code, all_individuals
        
        # 3. Ê¹ï¿½ï¿½Ô¤ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ð¿ï¿½ï¿½ï¿½É¸Ñ¡
        to_evaluate = []
        filtered_out = []
        
        if self.performance_predictor and self.code_history_pool:
            # ï¿½ï¿½È¡ï¿½ï¿½ï¿½Ðºï¿½Ñ¡ï¿½ï¿½ï¿½ï¿½ï¿½Ô¤ï¿½ï¿½ï¿½ï¿?
            predictions = []
            for candidate in candidates:
                prediction = self.predict_code_quality(candidate["embedding"])
                predictions.append((candidate, prediction[0], prediction[1]))
            
            # ï¿½ï¿½ï¿½ï¿½Ô¤ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Å¶ï¿½ï¿½ï¿½ï¿½ï¿½
            predictions.sort(key=lambda x: (-x[1], -x[2]))  # ï¿½ï¿½Ô¤ï¿½ï¿½ï¿½ï¿½(better)ï¿½ï¿½ï¿½ï¿½ï¿½Å¶È½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
            
            # ï¿½ï¿½ï¿½ï¿½eval_ratioÈ·ï¿½ï¿½Òªï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
            eval_count = max(1, int(len(candidates) * self.cfg.eval_ratio))
            logging.info(f"Will evaluate {eval_count} out of {len(candidates)} candidates based on eval_ratio {self.cfg.eval_ratio}")
            
            # Ñ¡ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ç°ï¿½Ä´ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿?
            for i, (candidate, is_better, confidence) in enumerate(predictions):
                if i < eval_count:
                    candidate["prediction_score"] = confidence
                    to_evaluate.append(candidate)
                    logging.info(f"Selected for evaluation: candidate {i+1} (confidence: {confidence:.4f})")
                else:
                    candidate["error"] = f"Filtered by predictor (rank: {i+1}, confidence: {confidence:.4f})"
                    filtered_out.append(candidate)
        else:
            # ï¿½ï¿½ï¿½Ã»ï¿½ï¿½Ô¤ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½eval_ratioï¿½ï¿½ï¿½Ñ¡ï¿½ï¿?
            eval_count = max(1, int(len(candidates) * self.cfg.eval_ratio))
            random.shuffle(candidates)  # ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ë³ï¿½ï¿?
            to_evaluate = candidates[:eval_count]
            filtered_out = candidates[eval_count:]
            for candidate in filtered_out:
                candidate["error"] = "Filtered by random selection"
        
        all_individuals.extend(filtered_out)
        
        # 4. ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ñ¡ï¿½ÐµÄ´ï¿½ï¿½ï¿½
        if to_evaluate:
            evaluated_batch = self._evaluate_parallel(to_evaluate, gen_dir)
            all_individuals.extend(evaluated_batch)
            
            # ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ñ½ï¿?
            successful = [ind for ind in evaluated_batch if ind.get("exec_success", False)]
            if successful:
                batch_best = min(successful, key=lambda x: x["obj"])
                if batch_best["obj"] < best_fitness:
                    best_fitness = batch_best["obj"]
                    best_code = batch_best["code"]
                    self._update_code_history_pool(batch_best)
        
        return best_fitness, best_code, all_individuals

    def evaluate_population(self, population: list[dict]) -> list[dict]:
        """Evaluate population using hybrid approach with updated logging"""
        if len(population) == 0:
            return []
        
        # For seed function or if no predictor available, evaluate all in parallel
        if self.elitist is None or self.performance_predictor is None:
            evaluated = self._evaluate_parallel(population)
            return evaluated
        
        # Get embeddings for all candidates (this is relatively fast)
        # Use 5th best global as threshold instead of elitist
        if len(self.best_individuals_overall) >= 5:
            threshold_individual = self.best_individuals_overall[4]
            threshold_embedding = self.get_code_embedding(threshold_individual["code"])
        else:
            threshold_embedding = self.get_code_embedding(self.elitist["code"])
        
        if threshold_embedding is None:
            evaluated = self._evaluate_parallel(population)
            return evaluated
        
        # Quick pre-filtering using predictor
        to_evaluate = []
        filtered_out = []
        predictions = []
        
        # Get predictions for all candidates
        for individual in population:
            individual["embedding"] = self.get_code_embedding(individual["code"])
            if individual["embedding"] is None:
                to_evaluate.append(individual)
                continue
            
            try:
                prediction = self.performance_predictor.predict_pair(
                    individual["embedding"],
                    threshold_embedding
                )
                predictions.append((individual, prediction['is_code1_better'], prediction['confidence']))
            except Exception as e:
                logging.error(f"Error in prediction: {e}")
                to_evaluate.append(individual)
                continue
        
        # Sort predictions by prediction result and confidence
        predictions.sort(key=lambda x: (-int(x[1]), -x[2]))  # Sort by is_better and confidence
        
        # Determine number of candidates to evaluate based on initial population or eval_ratio
        eval_ratio = 0.5 if self.iteration == 1 else self.cfg.eval_ratio
        eval_count = max(1, int(len(population) * eval_ratio))
        logging.info(f"Will evaluate {eval_count} out of {len(population)} candidates based on {'initial 0.5 ratio' if self.iteration == 1 else f'eval_ratio {self.cfg.eval_ratio}'}")
        
        # Select candidates for evaluation
        for i, (individual, is_better, confidence) in enumerate(predictions):
            if i < eval_count:
                individual["prediction_score"] = confidence
                to_evaluate.append(individual)
                logging.info(f"Selected for evaluation: candidate {i+1} (confidence: {confidence:.4f})")
            else:
                # Count filtered individuals in function_evals
                self.function_evals += 1
                individual = self.mark_invalid_individual(
                    individual,
                    f"Filtered by predictor (rank: {i+1}, confidence: {confidence:.4f})"
                )
                filtered_out.append(individual)
                logging.info(f"Code {individual['response_id']} filtered out by predictor")
        
        # Evaluate remaining candidates in parallel
        if to_evaluate:
            evaluated = self._evaluate_parallel(to_evaluate)
            
            # ÊÕ¼¯ÑµÁ·Êý¾Ý²¢¸üÐÂÔ¤²âÆ÷
            if self.performance_predictor is not None:
                for individual in evaluated:
                    if individual.get("exec_success", False) and individual.get("embedding") is not None:
                        # ÓëãÐÖµ´úÂë±È½Ï
                        self.performance_predictor.add_training_pair(
                            individual["embedding"],
                            threshold_embedding,
                            individual["obj"],
                            threshold_individual["obj"]
                        )
                        
                        # Óë¾«Ó¢´úÂë±È½Ï
                        if self.elitist is not None and self.elitist.get("embedding") is not None:
                            self.performance_predictor.add_training_pair(
                                individual["embedding"],
                                self.elitist["embedding"],
                                individual["obj"],
                                self.elitist["obj"]
                            )
            
            return evaluated + filtered_out
        return filtered_out

    def is_fitness_unique(self, fitness: float, tolerance: float = 1e-6) -> bool:
        """
        Check if fitness is unique (considering floating point errors)
        """
        for existing_fitness in self.unique_desc_results.keys():
            if abs(existing_fitness - fitness) < tolerance:
                return False
        return True

    def add_unique_description(self, description: str, fitness: float, code: str) -> bool:
        """
        Add new unique description to records
        Return if successfully added (if unique)
        """
        if self.is_fitness_unique(fitness):
            self.unique_desc_results[fitness] = {
                "description": description,
                "code": code
            }
            return True
        return False

    def short_term_reflection(self, desc1: dict, desc2: dict) -> str:
        """Generate short-term reflection comparing two descriptions with timing"""
        import time
        start_time = time.time()
        
        if desc1["obj"] == desc2["obj"]:
            raise ValueError("Descriptions have same fitness!")
        
        # Determine better and worse descriptions
        if desc1["obj"] < desc2["obj"]:
            better_desc, worse_desc = desc1, desc2
        else:
            better_desc, worse_desc = desc2, desc1
        
        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            problem_name=self.problem_name,
            worse_desc=worse_desc["description"],
            better_desc=better_desc["description"],
            worse_code=worse_desc["code"],
            better_code=better_desc["code"],
            worse_fitness=worse_desc["obj"],
            better_fitness=better_desc["obj"]
        )
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")
        return self.client.multi_chat_completion([messages])[0]

    def long_term_reflection(self, short_term_reflections: List[str]) -> None:
        """Update long-term reflection based on recent observations with timing"""
        import time
        start_time = time.time()
        
        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_name=self.problem_name,
            prior_reflection=self.long_term_reflection_str,
            new_reflection="\n".join(short_term_reflections)
        )
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        self.long_term_reflection_str = self.client.multi_chat_completion([messages])[0]
        
        # Save reflection to file with UTF-8 encoding
        gen_dir = self.output_dir / f"generation_{self.iteration}"
        with open(gen_dir / "short_term_reflections.txt", "w",encoding='utf-8') as f:
            f.write("\n".join(short_term_reflections))
        with open(gen_dir / "long_term_reflection.txt", "w",encoding='utf-8') as f:
            f.write(self.long_term_reflection_str)
        
        api_duration = time.time() - start_time
        self._log_timing(api_duration, "api")

    def evolve(self) -> Tuple[str, float, str]:
        """Modified evolution loop with reflection mechanisms"""
        logging.info("Starting evolution...")
        current_desc = self.initial_desc
        best_desc = current_desc
        best_fitness = float('inf')
        best_code = None
        
        # Track reflections for each generation
        short_term_reflections = []
        
        # Initialize global results tracking with fitness tracking
        global_results = {
            'generations': [],
            'fitness_seen': {}  # Track fitness values and their first occurrence
        }
        
        gen = 0
        while gen < self.num_generations:
            logging.info(f"Generation {gen + 1}/{self.num_generations}")
            self.iteration = gen
            
            # Create generation directory
            gen_dir = self.output_dir / f"generation_{gen}"
            gen_dir.mkdir(exist_ok=True)
            
            # Evaluate current description
            fitness, code, evaluated_individuals = self.evaluate_description(current_desc, gen_dir)
            
            # Save generation results
            self.save_generation_results(gen, gen_dir, current_desc, evaluated_individuals)
            
            # Update best solution if better
            if fitness < best_fitness:
                best_desc = current_desc
                best_fitness = fitness
                best_code = code
                
            # Track generation in global results, but only if fitness is new or better than existing
            current_fitness = float(fitness)
            should_add = True
            
            # Check if we've seen this fitness before
            if current_fitness in global_results['fitness_seen']:
                # If we've seen it, only keep if this is an earlier generation
                prev_gen = global_results['fitness_seen'][current_fitness]
                if gen >= prev_gen:
                    should_add = False
                    logging.info(f"Skipping generation {gen} as fitness {current_fitness} was already seen in generation {prev_gen}")
            
            if should_add:
                generation_result = {
                    'generation': gen,
                    'description': current_desc,
                    'best_fitness': current_fitness,
                    'best_code': code
                }
                global_results['generations'].append(generation_result)
                global_results['fitness_seen'][current_fitness] = gen
                
                # Clean up any previous generations with same fitness
                global_results['generations'] = [
                    g for g in global_results['generations']
                    if g['best_fitness'] != current_fitness or g['generation'] == gen
                ]
            
            # Generate next description using reflection-guided evolution
            if gen < self.num_generations - 1:
                # Find previous best generation for comparison
                # Only use generations that were kept after deduplication
                prev_gens = global_results['generations']
                if len(prev_gens) >= 2:
                    # Sort by fitness for selection
                    prev_gens.sort(key=lambda x: x['best_fitness'])
                    
                    # Select two different generations for reflection
                    better_gen = prev_gens[0]  # Best generation so far
                    worse_gen = prev_gens[-1]  # Worst generation so far
                    
                    # Create description objects for reflection
                    better_desc = {
                        "description": better_gen['description'],
                        "obj": better_gen['best_fitness'],
                        "code": better_gen['best_code']
                    }
                    worse_desc = {
                        "description": worse_gen['description'],
                        "obj": worse_gen['best_fitness'],
                        "code": worse_gen['best_code']
                    }
                    
                    # Generate reflection and add to history
                    reflection = self.short_term_reflection(better_desc, worse_desc)
                    short_term_reflections.append(reflection)
                    
                    # Periodic long-term reflection
                    if len(short_term_reflections) >= 3:
                        self.long_term_reflection(short_term_reflections)
                        short_term_reflections = []
                    
                    # Generate new description through crossover
                    current_desc = self.crossover(better_desc, worse_desc)
                    
                    # Occasionally mutate
                    if random.random() < self.mutation_rate:
                        variations = self.generate_variations(current_desc)
                        if variations:
                            current_desc = variations[0]
                else:
                    # For early generations, just mutate current description
                    variations = self.generate_variations(current_desc)
                    if variations:
                        current_desc = variations[0]
            
            gen += 1
        
        # Final long-term reflection if any remaining
        if short_term_reflections:
            self.long_term_reflection(short_term_reflections)
        
        # Before saving, clean up the fitness tracking dict as it's no longer needed
        del global_results['fitness_seen']
        
        # Sort generations by best_fitness before saving
        global_results['generations'].sort(key=lambda x: x['best_fitness'])
        
        # Save global results with UTF-8 encoding
        with open(self.output_dir / 'global_coevolve.json', 'w',encoding='utf-8') as f:
            json.dump(global_results, f, indent=2, ensure_ascii=False)
        
        # Update original YAML with best description using UTF-8 encoding
        self.problem_config['description'] = best_desc
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.problem_config, f, allow_unicode=True)
        
        # Update seed_func.txt with best code using UTF-8encoding
        seed_func_path = f"{self.root_dir}/prompts/{self.problem_name}/seed_func.txt"
        best_code_v1 = best_code.replace("heuristics_v2", "heuristics_v1")
        with open(seed_func_path, 'w', encoding='utf-8') as f:
            f.write(best_code_v1)
        
        logging.info(f"Evolution completed with {len(global_results['generations'])} generations")
        logging.info(f"Best fitness achieved: {best_fitness}")
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

        # Save successful solutions to results.json with UTF-8 encoding
        results = {
            'generation': generation,
            'description': description,
            'solutions': successful_solutions
        }

        with open(gen_dir / 'results.json', 'w',encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save failed solutions to results_err.json with UTF-8encoding
        error_results = {
            'generation': generation,
            'description': description,
            'failed_solutions': failed_solutions
        }

        with open(gen_dir / 'results_err.json', 'w',encoding='utf-8') as f:
            json.dump(error_results, f, indent=2, ensure_ascii=False)

        # Find best solution in this generation
        successful_individuals = [ind for ind in evaluated_individuals if ind.get("exec_success", False)]
        if successful_individuals:
            best_individual = min(successful_individuals, key=lambda x: x.get("obj", float('inf')))
        else:
            best_individual = evaluated_individuals[0]

        # Save best solution separately with UTF-8 encoding
        with open(gen_dir / 'best_solution.txt', 'w',encoding='utf-8') as f:
            f.write(f"Fitness: {best_individual.get('obj', float('inf'))}\n\n")
            f.write(f"Description:\n{description}\n\n")
            f.write(f"Code:\n{best_individual['code']}")

    def get_code_embedding(self, code: str) -> List[float]:
        """Get code embedding with caching"""
        # Use code hash as cache key
        code_hash = hash(code)
        
        # Check cache first
        if code_hash in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[code_hash]
        
        self.cache_misses += 1
        try:
            from utils.get_embeddings import llm_embedding_api
            embedding = llm_embedding_api(
                self.cfg.embedding_llm_model,
                self.cfg.embedding_end_point,
                self.cfg.embedding_api_key,
                code
            )
            if embedding is not None:
                self.embedding_cache[code_hash] = embedding
            return embedding
        except Exception as e:
            logging.error(f"Error getting code embedding: {e}")
            return None

    def _calculate_similarities(self, embedding: List[float], other_embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine similarities between one embedding and multiple others"""
        if not other_embeddings:
            return []
        
        # Convert to numpy arrays for vectorized computation
        embedding_array = np.array(embedding)
        other_embeddings_array = np.array(other_embeddings)
        
        # Calculate dot products
        dot_products = np.dot(other_embeddings_array, embedding_array)
        
        # Calculate norms
        embedding_norm = np.linalg.norm(embedding_array)
        other_norms = np.linalg.norm(other_embeddings_array, axis=1)
        
        # Calculate similarities
        similarities = dot_products / (embedding_norm * other_norms)
        
        return similarities.tolist()

    def _is_diverse_enough(self, new_embedding: List[float], existing_embeddings: List[List[float]], 
                          threshold: float = 0.95) -> bool:
        """Check if code is diverse enough compared to existing solutions"""
        if not existing_embeddings:
            return True
        
        # Calculate all similarities at once
        similarities = self._calculate_similarities(new_embedding, existing_embeddings)
        
        # Only reject if all similarities are above threshold
        return not all(sim > threshold for sim in similarities)

    def _update_population_with_diversity(self, evaluations: List[Tuple[float, dict]], 
                                        current_pop: List[dict]) -> List[dict]:
        """Update population with optimized diversity checking"""
        if not evaluations:
            return current_pop
        
        new_pop = current_pop.copy()
        
        # Extract all embeddings at once
        pop_embeddings = [ind["embedding"] for ind in new_pop]
        
        for fitness, individual in evaluations:
            # Always add if population not full
            if len(new_pop) < self.population_size:
                new_pop.append(individual)
                pop_embeddings.append(individual["embedding"])
                logging.info(f"Added individual to population (fitness: {fitness})")
                continue
            
            # Find worst individual
            worst_idx = max(range(len(new_pop)), key=lambda i: new_pop[i]["obj"])
            
            # Calculate similarities with all existing solutions at once
            similarities = self._calculate_similarities(individual["embedding"], pop_embeddings)
            max_similarity = max(similarities)
            
            # Replace if better than worst and not too similar
            if fitness < new_pop[worst_idx]["obj"] and max_similarity <= 0.98:
                logging.info(f"Replacing individual (fitness: {new_pop[worst_idx]['obj']}) with better one (fitness: {fitness})")
                new_pop[worst_idx] = individual
                pop_embeddings[worst_idx] = individual["embedding"]
            else:
                logging.debug(f"Individual not added (fitness:{fitness}, max similarity: {max_similarity:.4f})")
        
        return sorted(new_pop, key=lambda x: x["obj"])

    def _is_code_diverse(self, new_code: str, existing_codes: List[str], threshold: float = 0.92) -> bool:
        """Check if new code is sufficiently different from existing codes"""
        if not existing_codes:
            return True
        
        new_embedding = self.get_code_embedding(new_code)
        if new_embedding is None:
            return True
        
        existing_embeddings = []
        for code in existing_codes:
            embedding = self.get_code_embedding(code)
            if embedding is not None:
                existing_embeddings.append(embedding)
        
        if not existing_embeddings:
            return True
        
        # Calculate similarities with all existing codes at once
        similarities = self._calculate_similarities(new_embedding, existing_embeddings)
        
        # Code is diverse if it's not too similar to any existing code
        return not any(sim > threshold for sim in similarities)

    def predict_code_quality(self, code_embedding: List[float], threshold_fitness: float = None) -> Tuple[bool, float]:
        """Predict code quality using performance predictor"""
        if not self.performance_predictor or not self.code_history_pool:
            return True, 0.0
        
        # Select reference codes
        reference_codes = []
        if threshold_fitness is not None:
            # Use codes with better fitness as reference
            reference_codes = [code for code in self.code_history_pool 
                             if code['fitness'] < threshold_fitness]
            logging.info(f"Found {len(reference_codes)} referencecodes better than threshold {threshold_fitness}")
        
        # If no better codes found, use top performers
        if not reference_codes:
            reference_codes = sorted(self.code_history_pool, 
                                   key=lambda x: x['fitness'])[:3]
            logging.info("Using top 3 performers as reference codes")
        
        if not reference_codes:
            return True, 0.0
        
        # Get predictions against reference codes
        predictions = []
        for ref_code in reference_codes:
            try:
                pred = self.performance_predictor.predict_pair(
                    code_embedding,
                    ref_code['embedding']
                )
                predictions.append((pred['is_code1_better'], pred['confidence']))
                logging.debug(f"Prediction against reference code (fitness={ref_code['fitness']}): better={pred['is_code1_better']}, confidence={pred['confidence']:.4f}")
            except Exception as e:
                logging.error(f"Error in prediction: {e}")
                continue
        
        if not predictions:
            return True, 0.0
        
        # Calculate weighted confidence score
        positive_predictions = [conf for is_better, conf in predictions if is_better]
        if not positive_predictions:
            max_conf = max(conf for _, conf in predictions)
            logging.info(f"Code predicted to be worse than all references (max confidence: {max_conf:.4f})")
            return False, max_conf
        
        # Weight recent predictions more heavily
        weights = [1.0 + 0.1 * i for i in range(len(positive_predictions))]
        weighted_confidence = sum(w * c for w, c in zip(weights, positive_predictions)) / sum(weights)
        
        logging.info(f"Code predicted to be better with weighted confidence: {weighted_confidence:.4f}")
        return True, weighted_confidence

    def filter_code(self, code: str) -> str:
        """Filter and clean code for reflection"""
        try:
            # Remove comments
            lines = code.split('\n')
            filtered_lines = [line for line in lines if not line.strip().startswith('#')]
            
            # Remove empty lines at start and end
            while filtered_lines and not filtered_lines[0].strip():
                filtered_lines.pop(0)
            while filtered_lines and not filtered_lines[-1].strip():
                filtered_lines.pop()
            
            return '\n'.join(filtered_lines)
        except Exception as e:
            logging.error(f"Error filtering code: {e}")
            return code

    def _update_code_history_pool(self, individual: dict)-> None:
        """Update code history pool with new successful code"""
        history_entry = {
            'code': individual['code'],
            'embedding': individual['embedding'],
            'fitness': individual['obj'],
            'timestamp': datetime.now().timestamp()
        }
        
        # Add to pool
        self.code_history_pool.append(history_entry)
        
        # Keep pool size in check and sort by fitness and recency
        if len(self.code_history_pool) > self.max_history_pool_size:
            self.code_history_pool.sort(
                key=lambda x: (x['fitness'], -x['timestamp'])
            )
            # Keep best performing and most recent
            self.code_history_pool = self.code_history_pool[:self.max_history_pool_size]
            logging.info(f"Code history pool pruned to {len(self.code_history_pool)} entries")

    def _log_timing(self, duration: float, log_type: str):
        """Log timing information to appropriate file"""
        if log_type == "api":
            self.api_call_times.append(duration)
            log_path = self.api_log_path
            times_list = self.api_call_times
        elif log_type == "eval_batch":
            self.eval_batch_times.append(duration)
            log_path = self.eval_batch_log_path
            times_list = self.eval_batch_times
        else:  # eval_single
            self.eval_single_times.append(duration)
            log_path = self.eval_single_log_path
            times_list = self.eval_single_times
            
        total_time = sum(times_list)
        total_count = len(times_list)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Total count: {total_count}, Total time: {total_time:.2f}s\n")
            for t in times_list:
                f.write(f"{t:.4f}\n")

def main(cfg, root_dir, client):
    coevolver = DescriptionCoevolver(cfg, root_dir, client)
    best_desc, best_fitness, best_code = coevolver.evolve()
    logging.info(f"Best fitness achieved: {best_fitness}")
    return best_desc, best_fitness, best_code
