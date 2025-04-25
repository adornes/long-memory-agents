# Imports
import argparse
import asyncio
import json
import sys
import uuid

import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from rich.console import Console, group
from rich.panel import Panel
from rich.table import Table

# Parse terminal argument to choose similarity search type
parser = argparse.ArgumentParser(description="Choose similarity search type.")

parser.add_argument(
    "--similarity-search-threshold",
    type=float,
    default=0.75,
    help="Threshold of cosine similarity to return from similarity search.",
)

parser.add_argument(
    "--similarity-search-limit",
    type=int,
    default=5,
    help="Limit of results to return from similarity search.",
)

args = parser.parse_args()

# Initialize dotenv to load environment variables
load_dotenv()

# Initialize Rich for better output formatting and visualization
console = Console()

# Initialize the LLM
llm = init_chat_model("openai:gpt-4o-mini")

# Define the base URL for the API server
API_BASE_URL = "http://localhost:8000/v1"

# Create a single instance of AsyncClient to reuse
http_client = httpx.AsyncClient()


async def process_chunks(chunk, uuid_work, uuid_lead):
    """
    Asynchronously processes a chunk from the agent and displays information about tool calls or the agent's answer.

    Parameters:
        chunk (dict): A dictionary containing information about the agent's messages.

    Returns:
        None
    """
    if "agent" in chunk:
        for message in chunk["agent"]["messages"]:
            if "tool_calls" in message.additional_kwargs:
                tool_calls = message.additional_kwargs["tool_calls"]
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_arguments = json.loads(tool_call["function"]["arguments"])
                    tool_query = tool_arguments["query"]

                    console.print(
                        Panel.fit(
                            f"\nThe agent is calling the tool [bright_red]{tool_name}[/bright_red] with the query [bright_red]{tool_query}[/bright_red]. Please wait for the agent's answer...",
                            title="Tools",
                            border_style="red",
                        )
                    )
            else:
                agent_answer = message.content
                console.print(f"\nAgent:\n{agent_answer}", style="black on white")
                await persist_message(uuid_work, uuid_lead, "agent", agent_answer)


async def display_memory_retrieved(memory_retrieved):
    table = Table(title="Memory Retrieved")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column(
        f"Cosine Similarity (>{args.similarity_search_threshold})",
        justify="right",
        style="magenta",
    )
    table.add_column("Message", style="green")

    for i, memory in enumerate(memory_retrieved):
        table.add_row(
            str(i + 1),
            f"{memory['cosine_similarity']:.2f}",
            memory["message"],
        )

    console.print(table)


async def persist_message(uuid_work, uuid_lead, role, text):
    """
    Sends a request to the API endpoint to persist a message.

    Parameters:
        uuid_work (str): The UUID of the work associated with the message.
        uuid_lead (str): The UUID of the lead associated with the message.
        role (str): The role of the message sender (e.g., 'user', 'agent').
        text (str): The message text.

    Returns:
        None
    """
    try:
        response = await http_client.post(
            f"{API_BASE_URL}/persist_message",
            json={
                "uuid_work": uuid_work,
                "uuid_lead": uuid_lead,
                "role": role,
                "text": text,
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        console.print(f"Error persisting message: {e}", style="bold red")


async def retrieve_memory(
    message,
    similarity_search_threshold,
    similarity_search_limit,
    uuid_work,
    uuid_lead,
    verbose=False,
):
    """
    Sends a request to the API endpoint to retrieve memory.

    Parameters:
        message (str): The message text.
        similarity_search_threshold (float): The threshold of cosine similarity to return from similarity search.
        similarity_search_limit (int): The limit of results to return from similarity search.
        uuid_work (str): The UUID of the work associated with the message.
        uuid_lead (str): The UUID of the lead associated with the message.
        verbose (bool): Whether to print the similarity search results.

    Returns:
        list: A list of tuples containing the message and its cosine similarity.
    """
    try:
        response = await http_client.post(
            f"{API_BASE_URL}/retrieve_memory",
            json={
                "message_text": message,
                "similarity_search_threshold": similarity_search_threshold,
                "similarity_search_limit": similarity_search_limit,
                "uuid_work": uuid_work,
                "uuid_lead": uuid_lead,
            },
        )
        response.raise_for_status()
        memory_retrieved = response.json()["results"]

        if verbose and memory_retrieved:
            await display_memory_retrieved(memory_retrieved)

        return memory_retrieved
    except httpx.HTTPStatusError as e:
        console.print(f"Error retrieving memory: {e}", style="bold red")
        return []


async def display_agent_messages(messages):
    @group()
    def get_panels():
        for message in messages:
            if isinstance(message, SystemMessage):
                yield Panel.fit(
                    message.content,
                    title="System message",
                    border_style="deep_sky_blue1",
                )
            elif isinstance(message, HumanMessage):
                yield Panel.fit(
                    message.content,
                    title="Human message",
                    border_style="deep_sky_blue1",
                )

    console.print(
        Panel(
            get_panels(),
            title="Messages passed to the LangGraph agent:",
            border_style="bright_cyan",
        )
    )


async def main():
    uuid_work = input(
        "Enter the UUID of the work (or press Enter for a new UUID): "
    ) or str(uuid.uuid4())
    uuid_lead = input(
        "Enter the UUID of the lead (or press Enter for a new UUID): "
    ) or str(uuid.uuid4())

    console.print(
        Panel.fit(
            f"UUID of the work: [red]{uuid_work}[/red]\nUUID of the lead: [red]{uuid_lead}[/red]",
            title="UUIDs",
            border_style="magenta",
        )
    )

    tavily_tool = TavilySearchResults()
    langgraph_agent = create_react_agent(model=llm, tools=[tavily_tool])

    while True:
        user_question = input("\nUser:\n")

        if user_question.lower() == "quit":
            console.print("\nAgent:\nHave a nice day! :wave:\n", style="black on white")
            break

        memory_retrieved = await retrieve_memory(
            user_question,
            args.similarity_search_threshold,
            args.similarity_search_limit,
            uuid_work,
            uuid_lead,
            verbose=True,
        )

        messages = [HumanMessage(content=user_question)]
        memory_retrieved = [_["message"] for _ in memory_retrieved]

        if memory_retrieved:
            join_memory_retrieved = "\n".join(
                [f"- {message}" for message in memory_retrieved]
            )
            system_message = f"To answer the user's question, use this information which is part of the past conversation as a context:\n{join_memory_retrieved}"
            messages.insert(0, SystemMessage(content=system_message))

        await display_agent_messages(messages)

        async for chunk in langgraph_agent.astream({"messages": messages}):
            await process_chunks(chunk, uuid_work, uuid_lead)

    await http_client.aclose()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    finally:
        asyncio.run(http_client.aclose())  # Ensure the client is closed
