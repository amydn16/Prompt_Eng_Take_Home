import anthropic
import argparse
import json
import os
import requests
from dotenv import load_dotenv


load_dotenv()
model = os.getenv("ANTHROPIC_MODEL")

client = anthropic.Anthropic()

headers = {"User-Agent": "ClaudeWikiCLI/1.0 (amydn16@gmail.com)"}

BASIC_SYSTEM_PROMPT = """
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


def agent_loop(user_request: str, system_prompt: str, max_tokens: int = 1024):
    messages = [
        {
            "role": "user",
            "content": user_request,
        }
    ]

    try:
        response = client.messages.create(
            model=model or "",
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
                model=model or "",
                max_tokens=max_tokens,
                tools=tools,  # pyright: ignore
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=messages,  # pyright: ignore
            )

        final_text = next(block for block in response.content if block.type == "text")

        if search_used:
            print(
                f"I searched Wikipedia and came up with the following answer:\n{final_text.text}"
            )
        else:
            print(
                f"{final_text.text}\nI did not search Wikipdia to come up with this answer."
            )

    except Exception as e:
        print(f"Exited agent loop due to error: {e}")


def cli():
    parser = argparse.ArgumentParser(
        description="A tool that uses Claude to search Wikipedia and answer questions."
    )

    parser.add_argument("query", help="Query for Claude to answer")
    parser.add_argument(
        "--mode",
        choices=["default", "demo", "evals"],
        default="default",
        help="Choose default to submit a query, demo to see a demo, evals to run evals suite",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Optional argument setting max_tokens for the LLM, defaults to 1024",
    )

    args = parser.parse_args()

    if args.mode == "default" and not args.query:
        parser.error("A query is required in default mode")
    elif args.mode == "default":
        agent_loop(
            user_request=args.query,
            system_prompt=BASIC_SYSTEM_PROMPT,
            max_tokens=args.max_tokens,
        )
    elif args.mode == "demo":
        for demo_query in DEMO_QUERIES:
            agent_loop(
                user_request=demo_query,
                system_prompt=BASIC_SYSTEM_PROMPT,
                max_tokens=args.max_tokens,
            )
