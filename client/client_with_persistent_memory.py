# Imports
import argparse
import asyncio
import sys
import uuid

import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from rich.console import Console

# Parse terminal argument to choose similarity search type
parser = argparse.ArgumentParser(description="Choose similarity search type.")

parser.add_argument(
    "--similarity-search-threshold",
    default=0.75,
    required=False,
    help="Threshold of cosine similarity to return from similarity search."
)

parser.add_argument(
    "--similarity-search-limit",
    default=5,
    required=False,
    help="Limit of results to return from similarity search."
)

args = parser.parse_args()

# Initialize dotenv to load environment variables
load_dotenv()

# Initialize Rich for better output formatting and visualization
rich = Console()

# Initialize the LLM
llm = init_chat_model("openai:gpt-4o-mini")

# Initialize the embeddings model
embeddings = OpenAIEmbeddings()

# Define the base URL for the FastAPI server
API_BASE_URL = "http://localhost:8000"

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
                    rich.print(
                        f"\nThe agent is calling the tool [on deep_sky_blue1]{tool_name}[/on deep_sky_blue1] with the query [on deep_sky_blue1]{tool_query}[/on deep_sky_blue1]. Please wait for the agent's answer[deep_sky_blue1]...[/deep_sky_blue1]",
                        style="deep_sky_blue1"
                    )
            else:
                # If the message doesn't contain tool calls, extract and display the agent's answer

                # Extract the agent's answer
                agent_answer = message.content

                # Create the embedding vector for the agent's answer
                embeddings_response = embeddings.embed_query(agent_answer)

                # Retrieve the embedding vector for the agent's answer from the Embed LLM response
                agent_answer_embedding = embeddings_response

                # Insert the agent's answer and its embedding vector into the database
                await persist_message(uuid_work, uuid_lead, "agent", agent_answer, agent_answer_embedding)

                # Display the agent's answer
                rich.print(f"\nAgent:\n{agent_answer}", style="black on white")


async def persist_message(uuid_work, uuid_lead, role, text, embeddings):
    """
    Sends a request to the FastAPI endpoint to persist a message.

    Parameters:
        uuid_work (str): The UUID of the work associated with the message.
        uuid_lead (str): The UUID of the lead associated with the message.
        role (str): The role of the message sender (e.g., 'user', 'agent').
        text (str): The message text.
        embeddings (vector): The embedding vector of the message.

    Returns:
        None
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/persist_message", json={
            "uuid_work": uuid_work,
            "uuid_lead": uuid_lead,
            "role": role,
            "text": text,
            "embeddings": embeddings
        })
        response.raise_for_status()


async def similarity_search(message_embedding, similarity_search_threshold, similarity_search_limit, uuid_work, uuid_lead):
    """
    Sends a request to the FastAPI endpoint to perform a similarity search.

    Parameters:
        message_embedding (vector): The embedding vector of the message.
        similarity_search_threshold (float): The threshold of cosine similarity to return from similarity search.
        similarity_search_limit (int): The limit of results to return from similarity search.
        uuid_work (str): The UUID of the work associated with the message.
        uuid_lead (str): The UUID of the lead associated with the message.

    Returns:
        list: A list of tuples containing the message and its cosine similarity.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/similarity_search", json={
            "message_embedding": message_embedding,
            "similarity_search_threshold": similarity_search_threshold,
            "similarity_search_limit": similarity_search_limit,
            "uuid_work": uuid_work,
            "uuid_lead": uuid_lead
        })
        response.raise_for_status()
        return response.json()["results"]


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

    # Create a LangGraph agent
    langgraph_agent = create_react_agent(model=llm, tools=[])  # Tavily removed

    # Get the UUID of the work and the lead
    uuid_work = input("Enter the UUID of the work: ") or str(uuid.uuid4())
    uuid_lead = input("Enter the UUID of the lead: ") or str(uuid.uuid4())

    print(f"UUID of the work: {uuid_work}")
    print(f"UUID of the lead: {uuid_lead}")

    # Loop until the user chooses to quit the chat
    while True:
        # Get the user's question and display it in the terminal
        user_question = input("\nUser:\n")

        # Check if the user wants to quit the chat
        if user_question.lower() == "quit":
            rich.print(
                "\nAgent:\nHave a nice day! :wave:\n", style="black on white"
            )
            break

        # Create the embedding vector for the user's question
        embeddings_response = embeddings.embed_query(user_question)

        # Retrieve the embedding vector for the user's question from the Embed LLM response
        user_question_embedding = embeddings_response

        # Insert the user's question and its embedding vector into the database
        await persist_message(uuid_work, uuid_lead, "user", user_question, user_question_embedding)

        # Fetch the similarity search results
        similarity_search_results = await similarity_search(
            user_question_embedding,
            args.similarity_search_threshold,
            args.similarity_search_limit,
            uuid_work,
            uuid_lead
        )

        rich.print(
            "\n============================================================\n"
        )

        # Display all similarity search result messages
        # Those will be passed to the LangGraph agent as the system message
        rich.print(
            "[on deep_sky_blue1]Similarity search results:[/on deep_sky_blue1]",
            style="deep_sky_blue1"
        )

        rich.print(
            f"Here are the top {args.similarity_search_limit} most similar past messages with a cosine similarity equal to or greater than {args.similarity_search_threshold} to the latest user's question:",
            style="deep_sky_blue1"
        )

        for i, query_result in enumerate(similarity_search_results):
            rich.print(
                f"Message #{i+1} (cosine similarity = {query_result['cosine_similarity']:.2f}): {query_result['message']}",
                style="deep_sky_blue1"
            )

        # Prepare messages (i.e., human and system messages) to be passed to the LangGraph agent
        # Add the user's question to the HumanMessage object
        messages = [HumanMessage(content=user_question)]

        # Create a list to store the messages
        similar_messages = []

        # Iterate over the similarity search results and add them to the list
        for query_result in similarity_search_results:
            similar_messages.append(query_result["message"])

        # If there are similar messages returned from the similarity search, add them to the SystemMessage object
        if len(similar_messages) > 0:
            join_similar_messages = "\n".join(
                [f"- {message}" for message in similar_messages]
            )
            system_message = f"To answer the user's question, use this information which is part of the past conversation as a context:\n{join_similar_messages}"
            messages.insert(0, SystemMessage(content=system_message))

        # Initialize the system message and human message
        system_message = None
        human_message = None

        # Iterate over the messages and extract the system and human messages
        for message in messages:
            # Check if the message is a system message
            if isinstance(message, SystemMessage):
                system_message = message

            # Check if the message is a human message
            elif isinstance(message, HumanMessage):
                human_message = message

        # Display all messages that will be passed to the LangGraph agent (system and human messages)
        rich.print(
            "[on deep_sky_blue1]\nMessages passed to the LangGraph agent:[/on deep_sky_blue1]",
            style="deep_sky_blue1"
        )

        if system_message:
            rich.print(
                f"The system message:\n-----------------------\n{system_message.content}",
                style="deep_sky_blue1"
            )
        if human_message:
            rich.print(
                f"\nThe human message:\n-----------------------\n{human_message.content}",
                style="deep_sky_blue1"
            )

        rich.print("\n============================================================")

        # Use the async stream method of the LangGraph agent to get the agent's answer
        async for chunk in langgraph_agent.astream({"messages": messages}):
            # Process the chunks from the agent
            await process_chunks(chunk, uuid_work, uuid_lead)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the main async function
    asyncio.run(main())
