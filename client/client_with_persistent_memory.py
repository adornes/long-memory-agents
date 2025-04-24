# Imports
import argparse
import asyncio
import sys
import uuid

import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from rich.console import Console, group
from rich.panel import Panel
from rich.table import Table

# Parse terminal argument to choose similarity search type
parser = argparse.ArgumentParser(description="Choose similarity search type.")

parser.add_argument(
    "--similarity-search-threshold",
    default=0.75,
    required=False,
    help="Threshold of cosine similarity to return from similarity search.",
)

parser.add_argument(
    "--similarity-search-limit",
    default=5,
    required=False,
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


# Define an async function to process chunks from the agent
async def process_chunks(chunk, uuid_work, uuid_lead):
    """
    Asynchronously processes a chunk from the agent and displays information about tool calls or the agent's answer.

    Parameters:
        chunk (dict): A dictionary containing information about the agent's messages.

    Returns:
        None

    This function processes a chunk of data to check for agent messages asynchronously.
    It iterates over the messages and checks for tool calls.
    If a tool call is found, it extracts the tool name and query, then prints a formatted message using the Rich library.
    If no tool call is found, it extracts and prints the agent's answer using the Rich library.
    Additionally, it updates the database with the agent's response and logs the interaction.
    """

    # Check if the chunk contains an agent's message
    if "agent" in chunk:
        # Iterate over the messages in the chunk
        for message in chunk["agent"]["messages"]:
            # Check if the message contains tool calls
            if "tool_calls" in message.additional_kwargs:
                # If the message contains tool calls, extract and display an informative message with tool call details

                # Extract all the tool calls
                tool_calls = message.additional_kwargs["tool_calls"]

                # Iterate over the tool calls
                for tool_call in tool_calls:
                    # Extract the tool name
                    tool_name = tool_call["function"]["name"]

                    # Extract the tool query
                    tool_arguments = eval(tool_call["function"]["arguments"])
                    tool_query = tool_arguments["query"]

                    # Display an informative message with tool call details
                    console.print(
                        f"\nThe agent is calling the tool [on deep_sky_blue1]{tool_name}[/on deep_sky_blue1] with the query [on deep_sky_blue1]{tool_query}[/on deep_sky_blue1]. Please wait for the agent's answer[deep_sky_blue1]...[/deep_sky_blue1]",
                        style="deep_sky_blue1",
                    )
            else:
                # If the message doesn't contain tool calls, extract and display the agent's answer

                # Extract the agent's answer
                agent_answer = message.content

                # Display the agent's answer
                console.print(f"\nAgent:\n{agent_answer}", style="black on white")

                # Insert the agent's answer into the database
                await persist_message(uuid_work, uuid_lead, "agent", agent_answer)


async def display_memory_retrieved(memory_retrieved):
    # Display all memory retrieved messages
    # Those will be passed to the LangGraph agent as the system message

    # Create a Rich table to display memory retrieved
    table = Table(title="Memory Retrieved")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column(
        f"Cosine Similarity (>{args.similarity_search_threshold})",
        justify="right",
        style="magenta",
    )
    table.add_column("Message", style="green")

    # Add rows to the table for each memory retrieved
    for i, memory_retrieved in enumerate(memory_retrieved):
        table.add_row(
            str(i + 1),
            f"{memory_retrieved['cosine_similarity']:.2f}",
            memory_retrieved["message"],
        )

    # Print the table using Rich
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
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/persist_message",
            json={
                "uuid_work": uuid_work,
                "uuid_lead": uuid_lead,
                "role": role,
                "text": text,
            },
        )
        response.raise_for_status()


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
    async with httpx.AsyncClient() as client:
        response = await client.post(
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

        if verbose and len(memory_retrieved) > 0:
            await display_memory_retrieved(memory_retrieved)

        return memory_retrieved


async def display_agent_messages(messages):
    """
    Displays the system and human messages that will be passed to the LangGraph agent.

    Parameters:
        messages (list): A list of message objects (SystemMessage and HumanMessage).

    Returns:
        None
    """

    @group()
    def get_panels():
        # Iterate over the messages and print the system and human messages
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

    # Display all messages that will be passed to the LangGraph agent
    print("\n")
    console.print(
        Panel(
            get_panels(),
            title="Messages passed to the LangGraph agent:",
            border_style="bright_cyan",
        )
    )


# Define an async function to chat with the agent
async def main():
    """
    Entry point of the application. Initializes a persistent chat memory, creates a LangGraph agent, and handles user interaction in a loop until the user chooses to quit.

    Parameters:
        None

    Returns:
        None

    This function performs the following steps:
    1. Initializes a persistent chat memory.
    2. Creates a LangGraph agent with the specified model and tools.
    3. Enters a loop to interact with the user:
        - Prompts the user for a question.
        - Checks if the user wants to quit.
        - Uses the LangGraph agent to get the agent's answer.
        - Processes the chunks from the agent.

    The function supports two options for similarity search:
    - "limit": Returns the top 5 most similar past messages that have the highest cosine similarity to the latest user message.
    - "threshold": Returns all past messages that have a cosine similarity equal to or greater than 0.75 to the latest user message.
    """

    # Get the UUID of the work and the lead
    uuid_work = input("Enter the UUID of the work: ") or str(uuid.uuid4())
    uuid_lead = input("Enter the UUID of the lead: ") or str(uuid.uuid4())

    print("\n")

    # Display the UUIDs in a panel
    console.print(
        Panel.fit(
            f"UUID of the work: [red]{uuid_work}[/red]\nUUID of the lead: [red]{uuid_lead}[/red]",
            title="UUIDs",
            border_style="magenta",
        )
    )

    # Create a LangGraph agent
    langgraph_agent = create_react_agent(model=llm, tools=[])  # Tavily removed

    # Loop until the user chooses to quit the chat
    while True:
        # Get the user's question and display it in the terminal
        user_question = input("\nUser:\n")

        # Check if the user wants to quit the chat
        if user_question.lower() == "quit":
            console.print("\nAgent:\nHave a nice day! :wave:\n", style="black on white")
            break

        # Fetch the memory retrieved
        memory_retrieved = await retrieve_memory(
            user_question,
            args.similarity_search_threshold,
            args.similarity_search_limit,
            uuid_work,
            uuid_lead,
            verbose=True,
        )

        # Prepare messages (i.e., human and system messages) to be passed to the LangGraph agent
        # Add the user's question to the HumanMessage object
        messages = [HumanMessage(content=user_question)]

        # Create a list to store the messages
        memory_retrieved = [_["message"] for _ in memory_retrieved]

        # If there are similar messages returned from the similarity search, add them to the SystemMessage object
        if len(memory_retrieved) > 0:
            join_memory_retrieved = "\n".join(
                [f"- {message}" for message in memory_retrieved]
            )
            system_message = f"To answer the user's question, use this information which is part of the past conversation as a context:\n{join_memory_retrieved}"
            messages.insert(0, SystemMessage(content=system_message))

        # Use the new function to prepare and display agent messages
        await display_agent_messages(messages)

        # Use the async stream method of the LangGraph agent to get the agent's answer
        async for chunk in langgraph_agent.astream({"messages": messages}):
            # Process the chunks from the agent
            await process_chunks(chunk, uuid_work, uuid_lead)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the main async function
    asyncio.run(main())
