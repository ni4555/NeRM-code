import hydra
import logging
import os
from pathlib import Path
import subprocess
from utils.utils import init_client
import yaml

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


def reload_config(cfg, problem_name):
    """reload config yaml"""
    yaml_path = os.path.join(ROOT_DIR, "cfg", "problem", f"{problem_name}.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            problem_config = yaml.safe_load(f)

        # update problem setting in cfg
        cfg.problem.description = problem_config.get('description', cfg.problem.description)

        logging.info(f"Reloaded configuration from {yaml_path}")
    else:
        logging.warning(f"Configuration file {yaml_path} not found")
    return cfg


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    client = init_client(cfg)

    if cfg.algorithm == "coevolve":
        logging.info(f"Using Description Reflection: {cfg.use_reflection}")
        logging.info(f"Using Code Predictor: {cfg.use_predictor}")
        logging.info(f"Using Serial Mode: {cfg.use_serial}")
        logging.info(f"Using Finetune Predictor: {cfg.get('finetune_predictor', False)}")

        if cfg.get("use_serial", False):
            if cfg.get("use_reflection", False) and cfg.get("use_predictor", False):
                logging.info(f"Loading Serial Coevolve and ReEvo with reflect and predictor")
                from coevolve_reflect_with_predictor_serial import DescriptionCoevolver
                from reevo_predictor_serial import ReEvo as LHH
            elif cfg.get("use_reflection", False):
                logging.info(f"Loading Serial Coevolve with reflect only and Serial ReEvo")
                from coevolve_reflect_serial import DescriptionCoevolver
                from reevo_serial import ReEvo as LHH
            else:
                logging.info(f"Loading Serial pure Coevolve and ReEvo")
                from coevolve_serial import DescriptionCoevolver
                from reevo_serial import ReEvo as LHH
        else:
            if cfg.get("use_reflection", False):
                if cfg.get("use_predictor", False):
                    if cfg.get("finetune_predictor", False):
                        logging.info(f"Loading Coevolve and ReEvo with reflect and finetuned predictor")
                        from coevolve_reflect_with_predictor_finetune import DescriptionCoevolver
                        from reevo_predictor_finetune import ReEvo as LHH
                    else:
                        logging.info(f"Loading Coevolve and ReEvo with reflect and predictor")
                        from coevolve_reflect_with_predictor import DescriptionCoevolver
                        from reevo_predictor import ReEvo as LHH
                else:
                    logging.info(f"Loading Coevolve with reflect only and original ReEvo")
                    from coevolve_reflect import DescriptionCoevolver
                    from reevo import ReEvo as LHH
            else:
                if cfg.get("use_predictor", False):
                    logging.info(f"Loading Coevolve and ReEvo with predictor only but without reflection")
                    from coevolve_with_predictor import DescriptionCoevolver
                    from reevo_woreflect_predictor import ReEvo as LHH
                else:
                    logging.info(f"Loading pure Coevolve and ReEvo")
                    from coevolve import DescriptionCoevolver
                    from reevo import ReEvo as LHH

        coevolver = DescriptionCoevolver(cfg, ROOT_DIR, client)
        best_desc, best_fitness, best_code = coevolver.evolve()

        logging.info(f"Best description: {best_desc}")
        logging.info(f"Best description fitness: {best_fitness}")

        # reload config
        cfg = reload_config(cfg, cfg.problem.problem_name)

    elif cfg.algorithm == "moh":
        logging.info("Loading MoH (Metamorphic heuristic) algorithm")
        from moh import MoH as LHH

        # Initialize and run MoH
        lhh = LHH(cfg, ROOT_DIR, client)
        best_desc, best_fitness, best_code = lhh.evolve()

        logging.info(f"Best description: {best_desc}")
        logging.info(f"Best fitness: {best_fitness}")

        # Update config with best description
        cfg = reload_config(cfg, cfg.problem.problem_name)

        # Save best code for validation
        best_code_overall = best_code
        best_code_path_overall = None  # MoH doesn't use code paths in the same way

    elif cfg.algorithm == "reevo":
        if cfg.get("use_serial", False):
            logging.info(f"Loading Serial ReEvo")
            if cfg.get("use_predictor", False):
                logging.info(f"with predictor")
                from reevo_predictor_serial import ReEvo as LHH
            else:
                from reevo_serial import ReEvo as LHH
        else:
            if cfg.get("use_predictor", False):
                if cfg.get("use_reflection", False):
                    if cfg.get("finetune_predictor", False):
                        logging.info(f"Loading ReEvo with finetuned predictor")
                        from reevo_predictor_finetune import ReEvo as LHH
                    else:
                        logging.info(f"Loading ReEvo with predictor")
                        from reevo_predictor import ReEvo as LHH
                else:
                    logging.info(f"Loading ReEvo without reflection but with predictor")
                    from reevo_woreflect_predictor import ReEvo as LHH
            else:
                logging.info(f"Use original ReEvo")
                from reevo import ReEvo as LHH
    elif cfg.algorithm == "no_revolution":
        logging.info("Loading NoRevolution Generator (ablation study)")
        from ablation_no_revolution import NoRevolutionGenerator as LHH
    elif cfg.algorithm == "ael":
        from baselines.ael.ga import AEL as LHH
    elif cfg.algorithm == "eoh":
        from baselines.eoh import EoH as LHH
    else:
        raise NotImplementedError

    # For algorithms other than MoH, run the original evolution
    if cfg.algorithm != "moh":
        lhh = LHH(cfg, ROOT_DIR, client)
        best_code_overall, best_code_path_overall = lhh.evolve()
        logging.info(f"Best Code Overall: {best_code_overall}")
        logging.info(f"Best Code Path Overall: {best_code_path_overall}")

    # Run validation
    with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py", 'w') as file:
        file.writelines(best_code_overall + '\n')
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script...: {test_script}")
    with open(test_script_stdout, 'w') as stdout:
        subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
    logging.info(f"Validation script finished. Results are saved in {test_script_stdout}.")

    # Print the results
    with open(test_script_stdout, 'r') as file:
        for line in file.readlines():
            logging.info(line.strip())


if __name__ == "__main__":
    main()