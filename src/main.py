import anthropic
import json
import os
import requests
from dotenv import load_dotenv


load_dotenv()
generator_model = os.getenv("ANTHROPIC_GENERATOR_MODEL")

client = anthropic.Anthropic()

headers = {"User-Agent": "ClaudeWikiCLI/1.0 (amydn16@gmail.com)"}

GENERATOR_SYSTEM_PROMPT = """
You are a helpful assistant that provides helpful, clear, and concise answers, using Wikipedia as efficiently as possible when necessary and appropriate.
"""

DEMO_QUERIES = [
    "When was the very first game of basketball played?",
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


def get_wikipedia_page(query: str):
    search_res = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        },
        headers=headers,
    )
    best_title = search_res.json()["query"]["search"][0]["title"].replace(" ", "_")

    res = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{best_title}",
        headers=headers,
    )
    content = res.json()["extract"]

    return content


def run_tool(name, tool_input):
    if name == "search_wikipedia":
        return {"wikipedia_page_content": tool_input["query"]}
    return {"error": f"Unknown tool: {name}"}


def agent_loop(query: str, system_prompt: str, max_tokens: int):
    print(f"System prompt for agent:\n{system_prompt}")
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

        search_used = False
        if response.stop_reason == "tool_use":
            search_used = True

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
                max_tokens=max_tokens,
                tools=tools,  # pyright: ignore
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=messages,  # pyright: ignore
            )

        final_text = next(block for block in response.content if block.type == "text")

        if search_used:
            result = f"I searched Wikipedia and came up with the following answer: {final_text.text}"
        else:
            result = f"I did not search Wikipedia to come up with this answer: {final_text.text}"

        print(result)
        return result
    except Exception as e:
        print(f"Exited agent loop due to error: {e}")


def agent_loop_query_list(query_list: list[str], system_prompt: str, max_tokens: int):
    print(f"Query list: {query_list}")
    if not query_list:
        raise Exception("Query list cannot be empty")

    return [agent_loop(query, system_prompt, max_tokens) for query in query_list]
