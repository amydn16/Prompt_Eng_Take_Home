import anthropic
import json
import os
from dotenv import load_dotenv
from src.main import agent_loop


load_dotenv()
grader_model = os.getenv("ANTHROPIC_GRADER_MODEL")

client = anthropic.Anthropic()

GRADER_SYSTEM_PROMPT = """
You are a helpful assistant that grades an answer to a query against an example of a correct answer. Do not take answer length or writing style into consideration. Only consider whether the answer being graded aligns with the correct answer example in terms of (1) whether a Wikipedia search was done and (2) the answer to the query. 
"""


def run_agent_on_evals(file_path: str, generator_system_prompt: str, max_tokens: int):
    try:
        with open(file_path, "r") as f:
            evals_data = json.load(f)

        for i in range(len(evals_data)):
            model_answer = agent_loop(
                evals_data[i]["question"], generator_system_prompt, max_tokens
            )
            evals_data[i]["model_answer"] = model_answer

        return evals_data

    except Exception as e:
        print(f"Error occurred during evals: {e}")
        return []


def llm_grader(
    grader_system_prompt: str,
    query: str,
    correct_answer: str,
    model_answer: str,
    max_tokens: int = 1024,
):
    grading_context = f"Query: {query}\nExample correct answer: {correct_answer}\nAnswer to grade: {model_answer}"

    print(f"Grader system prompt:\n{grader_system_prompt}")
    print(f"Grading context:\n{grading_context}")

    messages = [
        {
            "role": "user",
            "content": grading_context,
        }
    ]

    try:
        response = client.messages.create(
            model=grader_model or "",
            system=grader_system_prompt,
            max_tokens=max_tokens,
            tools=[],  # pyright: ignore
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=messages,  # pyright: ignore
        )

        final_text = next(block for block in response.content if block.type == "text")

        score = float(final_text.text.strip())
        print(f"Score: {score}")
        return score

    except Exception as e:
        print(f"Error occurred during LLM grading of evals results: {e}")
        return


def full_agent_evals(
    file_path: str,
    generator_system_prompt: str,
    grader_system_prompt: str,
    max_tokens: int = 1024,
):
    evals_data = run_agent_on_evals(file_path, generator_system_prompt, max_tokens)

    score = 0.0
    for i in range(len(evals_data)):
        if new_score := llm_grader(
            grader_system_prompt,
            evals_data[i]["question"],
            evals_data[i]["sample_correct_answer"],
            evals_data[i]["model_answer"],
        ):
            score += new_score

    print(f"Final score: {score} ({100 * score / len(evals_data)}% correct)")
    return score
