# claude-wikipedia CLI

A simple CLI that uses Claude and the Wikipedia API to answer a natural language query. It takes a natural language query and sends it to a Claude model. Claude then chooses whether to search Wikipedia to answer the query before issuing an answer. The answer always indicates whether the information in it comes from a Wikipedia search or from Claude's internal knowledge.

I used Claude Sonnet 4.6 to answer queries. The project directory also includes an evals suite of more than 20 open-ended queries that test Claude's ability to answer questions by searching Wikipedia and, to a lesser extent, by using its internal knowledge. I used Claude Haiku 4.5 as the grader for evals.

## Requirements

- Python >=3.12
- [uv](https://github.com/astral-sh/uv) (installed for dependency management)
- Anthropic API key

## Installation

1. Clone the repository (only Robert Ying has access):

```bash
git clone https://github.com/amydn16/Prompt_Eng_Take_Home.git
cd Prompt_Eng_Take_Home
```

2. Install dependencies from `uv.lock` and create a new virtual environment at `.venv`:

```bash
uv sync --all-groups
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Setup

1. Create a `.env` file from the provided `.env.example`:

```bash
cp .env.example .env
```

2. Add the following to `.env.example`:

    - `ANTHROPIC_API_KEY`: Your Anthropic API key
    - `ANTHROPIC_GENERATOR_MODEL`: The model name for the Claude model answering queries (e.g., claude-sonnet-4-6)
    - `ANTHROPIC_GRADER_MODEL`: The model name for the Claude model grading evals performance

3. Export the above as environment variables:

```bash
source .env
```

## Usage

### Running a demo

Use demo mode to see Claude answer 2 simple sample queries:

```bash
claude-wikipedia --mode demo
```

The CLI always prints to the terminal the query, the system prompt for the query-answering model, and the summary of any Wikipedia page found, in addition to the answer to the query.

### Submitting a query

Send a natural language query to the CLI to get an answer from Claude, wrapping any query containing spaces in single quotes:

```bash
claude-wikipedia 'who is the current governor of New York?'
```

### Running evals

Run the query-answering model on a subset of the evals suite containing just 2 queries:

```bash
claude-wikipedia --mode evals-mini
```

In evals mode, the CLI always prints to the terminal the system prompt for the grader model, the query, an example of a correct answer, the query-answering model's answer, the numerical score that the grader model gives the answer, and any rationale for the score.

Run the query-answering model on the full evals suite containing more than 20 queries:

```bash
claude-wikipedia --mode evals-full
```