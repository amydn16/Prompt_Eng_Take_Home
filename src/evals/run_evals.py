import anthropic
import json
import os
from dotenv import load_dotenv
from src.main import agent_loop


load_dotenv()
grader_model = os.getenv("ANTHROPIC_GRADER_MODEL")

client = anthropic.Anthropic()

GRADER_SYSTEM_PROMPT = """
You are a helpful assistant that grades a response to a query against an example of a correct response and returns a number as the grade. Each response, both that to be graded and the correct example, is one sentence containing a colon. The independent clause before the colon will always indicate whether a Wikipedia search was done to form the part of the sentence after the colon, which is the actual answer to the query.

When grading a response, do not take the length or writing style of the query answer into consideration. Only consider whether the response aligns with the correct response example in terms of (1) whether a Wikipedia search was done and (2) whether the query answer is complete and correct. Consider the query answer in the correct response example to be your one and only source of truth for grading a response. Do not use your internal knowledge or any context beyond the query answer in the correct response example to grade a response.

Specifically, if the correct response example says that a Wikipedia search was done to form its query answer, but the response being graded says that a Wikipedia search was not done to form its query answer, return 0. Conversely, if the correct response example says that a Wikipedia search was not done to form its query answer, but the response being graded says that a Wikipedia search was done to form its query answer, return 0 as well.

If both the correct response example and the response being graded had a Wikipedia search done to form their respective query answers, but the latter provides an incorrect or an incomplete query answer compared to the former, return 0.5. Similarly, if both the correct response example and the response being graded did not have a Wikipedia search done to form their respective query answers, but the latter provides an incorrect or an incomplete query answer compared to the former, also return 0.5. Here, "incomplete" means that the query answer lacked some detail that plays arguably a considerable role in answering the query thoroughly.

If both the correct response example and the response being graded had a Wikipedia search done to form their respective query answers, and the latter provides an equivalently complete and correct query answer as the former, return 1. Similarly, if both the correct response example and the response being graded did not have a Wikipedia search done to form their respective query answers, and the latter provides an equivalently complete and correct query answer as the former, return 1. It's perfectly fine if the response being graded provides a more verbose or detailed query answer than the correct response example, so long as it conveys the same idea as the correct response example. In this case, you should still return 1.

Always remember to return the number that is your grade for the response. You do not need to return your rationale for the grade, but if you want to do so, you MUST format your return as '[grade]\\n[rationale]'.
"""


def run_agent_on_evals(file_path: str, generator_system_prompt: str, max_tokens: int):
    try:
        with open(file_path, "r") as f:
            evals_data = json.load(f)

        for i in range(len(evals_data)):
            model_response = agent_loop(
                evals_data[i]["question"], generator_system_prompt, max_tokens
            )
            evals_data[i]["model_response"] = model_response

        return evals_data

    except Exception as e:
        print(f"Error occurred during evals: {e}")
        return []


def llm_grader(
    grader_system_prompt: str,
    query: str,
    correct_response: str,
    model_response: str,
    max_tokens: int,
):
    grading_context = f"Query: {query}\nExample correct answer: {correct_response}\nAnswer to grade: {model_response}"
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

        final_response = next(block for block in response.content if block.type == "text")
        
        final_response_parts = final_response.text.split("\n")
        score = float(final_response_parts[0].strip())
        rationale = ""
        if len(final_response_parts) >= 2:
            rationale += " ".join(final_response_parts[1:])

        print(f"Score: {score}")
        print(f"Grading rationale (if any): {rationale}")
        return score

    except Exception as e:
        print(f"Error occurred during LLM grading of evals results: {e}")
        return


def full_agent_evals(
    file_path: str,
    generator_system_prompt: str,
    grader_system_prompt: str,
    max_tokens: int,
):
    print(f"Grader system prompt:\n{grader_system_prompt}")

    evals_data = run_agent_on_evals(file_path, generator_system_prompt, max_tokens)

    score = 0.0
    for i in range(len(evals_data)):
        if new_score := llm_grader(
            grader_system_prompt,
            evals_data[i]["question"],
            evals_data[i]["sample_correct_response"],
            evals_data[i]["model_response"],
            max_tokens,
        ):
            score += new_score

    print(f"Final score: {score} ({100 * score / len(evals_data)}% correct)")
    return score
