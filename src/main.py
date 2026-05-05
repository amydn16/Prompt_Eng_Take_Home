import anthropic
import json
import os
import random
import requests
import time
import wikipedia
from dotenv import load_dotenv


load_dotenv()
generator_model = os.getenv("ANTHROPIC_GENERATOR_MODEL")

client = anthropic.Anthropic()

headers = {"User-Agent": "ClaudeWikiCLI/1.0 (amydn16@gmail.com)"}

GENERATOR_SYSTEM_PROMPT = """
You are a helpful assistant that provides helpful, clear, and concise responses, using Wikipedia as efficiently as possible when necessary and appropriate. When you receive a query, the first thing you must always do is consider whether you need to search Wikipedia to form an answer. E.g., there are stable, well-known facts in science and history that are widely accepted as true and do/can not change. In contrast, there are facts that naturally need to be updated regularly. There may also exist facts that are not available on Wikipedia. Always consult these guidelines and think carefully to decide whether to search Wikipedia to answer a query. Then when composing your response, you MUST format it as: 

'[Independent clause indicating whether you searched Wikipedia and whether you used information from Wikipedia or from your internal knowledge to form the answer to the query]: [answer to the query]'.

ALWAYS REMEMBER: if you do not find the information needed to answer a query from searching Wikipedia, and you decide to use your internal knowledge to form the answer, you MUST clearly indicate in your response (in the independent clause before the colon) that you were not able to find the information you needed from Wikipedia and are using your internal knowledge to form the answer.
"""

DEMO_QUERIES = [
    "When was basketball invented?",
    "Who's the current governor of Virginia?",
]

tools = [
    {
        "name": "search_wikipedia",
        "description": "Search Wikipedia to find the answer to a natural language query",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language query, such as 'Who discovered DNA'",
                },
            },
            "required": ["query"],
        },
    }
]


def get_wikipedia_page(query: str, retries: int = 10) -> str:
    wikipedia.set_lang("en")

    for attempt in range(retries):
        print(f"Trying to get Wikipedia page for query '{query}', attempt #{attempt}")
        try:
            page = wikipedia.page(query, auto_suggest=True)
            print(f"Summary of Wikipedia page found: {page.summary}")
            return page.summary
        except wikipedia.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0])
                print(f"Summary of first Wikipedia page found: {page.summary}")
                return page.summary
            except Exception:
                return "Disambiguation could not be resolved."
        except wikipedia.PageError:
            return "No matching Wikipedia article found."
        except Exception as e:
            if attempt < retries - 1:
                delay = (2**attempt) + random.uniform(0, 0.5)
                print(f"Sleeping for {delay} s... due to exception {e}")
                time.sleep(delay)
            else:
                return f"Wikipedia lookup failed after {retries} attempts: {str(e)}"

    return "Unknown error in get_wikipedia_page."


def run_tool(name: str, tool_input: dict) -> dict:
    if name == "search_wikipedia":
        return {"wikipedia_page_content": get_wikipedia_page(tool_input["query"])}
    return {"error": f"Unknown tool: {name}"}


def agent_loop(query: str, system_prompt: str, max_tokens: int):
    print(f"Agent system prompt:\n{system_prompt}")
    print(f"User: {query}")

    messages = [
        {
            "role": "user",
            "content": query,
        }
    ]

    try:
        response = client.messages.create(
            model=generator_model or "",
            system=system_prompt,
            max_tokens=max_tokens,
            tools=tools,  # pyright: ignore
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=messages,  # pyright: ignore
        )

        while response.stop_reason == "tool_use":
            tool_use = next(
                block for block in response.content if block.type == "tool_use"
            )
            result = run_tool(tool_use.name, tool_use.input)

            messages.append({"role": "assistant", "content": response.content})  # pyright: ignore
            messages.append(
                {
                    "role": "user",
                    "content": [  # pyright: ignore
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": json.dumps(result),
                        }
                    ],
                }
            )

            response = client.messages.create(
                model=generator_model or "",
                system=system_prompt,
                max_tokens=max_tokens,
                tools=tools,  # pyright: ignore
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=messages,  # pyright: ignore
            )

        final_text = next(block for block in response.content if block.type == "text")

        result = final_text.text
        print(f"Response: {result}")
        return result
    except Exception as e:
        print(f"Exited agent loop due to error: {e}")


def agent_loop_query_list(query_list: list[str], system_prompt: str, max_tokens: int):
    print(f"Query list: {query_list}")

    if not query_list:
        raise Exception("Query list cannot be empty")

    return [agent_loop(query, system_prompt, max_tokens) for query in query_list]
