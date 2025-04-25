Add Tavily
Try Pareto for improving/refactoring the client script

Check alternatives and compare:
- https://www.getzep.com/
- https://docs.mem0.ai/platform/overview

AWS Architecture: https://tess.pareto.io/share/4df03db8-e294-4e18-9cc7-270aae100f7e

# 1. Knowledge Graph Data models

Nodes/Entities = Person, Concept, Event, Location
Edge/Paths/Relationships, distance
Clusters/Communities


# 2. Query the Knowledge Graph

PathRAG
    PathRAG paper
    https://github.com/BUPT-GAMMA/PathRAG/tree/main
    Graph RAG Evolved: PathRAG (Relational Reasoning Paths): https://www.youtube.com/watch?v=oetP9uksUwM

Message-based similarity search should be used when no prior knowledge is found (graph is empty).

Entities and relationships are used to build a Relational Path: "Plants -> (are infested by) -> Aphids -> (are fed on by) -> Ladybugs

The Relational Path above is then used as context for the agent.


# 3. Script to populate the Knowledge Graph

Build basic script to upsert data to the Knowledge Graph.

========================================================================================

Chunking Strategy:
1. Character/Token Based Chunking
2. Recursive Character/Token Based Chunking +++
3. Semantic Chunking
4. Cluster Semantic Chunking
5. LLM Semantic Chunking

https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

"Sentence chunking"

Late-chunking:

    https://www.youtube.com/watch?v=Hj7PuK1bMZU
    https://colab.research.google.com/drive/15vNZb6AsU7byjYoaEtXuNu567JWNzXOz
    https://jina.ai/news/late-chunking-in-long-context-embedding-models/
    https://jina.ai/news/what-late-chunking-really-is-and-what-its-not-part-ii/
    https://github.com/jina-ai/late-chunking?tab=readme-ov-file
    https://arxiv.org/abs/2409.04701
    https://weaviate.io/blog/late-chunking

PathRAG paper - Pg 3

Semantic embeddings given the broader context.



# 4. Automate script execution (daily?)

Schedule the script with an orchestrator.


# 5. Refine it all

Review overall advanced RAG techniques and refine the project:

https://www.datacamp.com/blog/rag-advanced
https://weaviate.io/ebooks/advanced-rag-techniques
https://www.godaddy.com/resources/news/llm-from-the-trenches-10-lessons-learned-operationalizing-models-at-godaddy
AI Engineering Chip Huyen - https://www.youtube.com/watch?v=JV3pL1_mn2M&list=WL&index=3
