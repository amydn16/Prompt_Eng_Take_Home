import json
from src.main import agent_loop_query_list


def run_agent_on_evals(file_path: str, system_prompt: str, max_tokens: int):
    with open(file_path, "r") as f:
        evals_data = json.load(f)

    queries = list(evals_data.keys())
    answers = agent_loop_query_list(queries, system_prompt, max_tokens)


def mc_grader():
    pass


def llm_grader():
    pass


def full_agent_eval_loop(file_path: str, system_prompt: str, max_tokens: int = 1024):
    pass
