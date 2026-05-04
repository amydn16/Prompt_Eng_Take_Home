import argparse
from src.main import (
    BASIC_SYSTEM_PROMPT,
    DEMO_QUERIES,
    agent_loop,
    agent_loop_query_list,
)


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
            query=args.query,
            system_prompt=BASIC_SYSTEM_PROMPT,
            max_tokens=args.max_tokens,
        )
    elif args.mode == "demo":
        agent_loop_query_list(
            query_list=DEMO_QUERIES,
            system_prompt=BASIC_SYSTEM_PROMPT,
            max_tokens=args.max_tokens,
        )
