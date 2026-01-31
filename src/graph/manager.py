import os
from dotenv import load_dotenv
from datetime import datetime
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver

from src.utils.config_loader import CONFIG

load_dotenv()


class GraphitiManager:
    def __init__(self):
        # 1. Configure the Local LLM (Ollama)
        llm_config = LLMConfig(
            api_key="ollama", 
            model="sciphi/triplex:latest", 
            small_model="sciphi/triplex:latest",
            base_url="http://localhost:11434/v1",
        )
        llm_client = OpenAIGenericClient(config=llm_config)

        # 2. Configure Local Embedder (Ollama)
        embedder_config = OpenAIEmbedderConfig(
            api_key="ollama",
            embedding_model="nomic-embed-text:latest",
            embedding_dim=768,
            base_url="http://localhost:11434/v1",
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        # 3. Initialize Graphiti with Cardiology Context
        URI = os.getenv("neo4j_uri")
        USERNAME = os.getenv("neo4j_username")
        PASSWORD = os.getenv("neo4j_password")

        if URI is None or USERNAME is None or PASSWORD is None:
            raise ValueError("Missing required Neo4j environment variables: neo4j_uri, neo4j_username, or neo4j_password")

        driver = Neo4jDriver(
            uri=URI,
            user=USERNAME,
            password=PASSWORD,
        )
        self.graph = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=OpenAIRerankerClient(client=llm_client.client, config=llm_config)
            )

    async def ingest_clinical_episode(self, text: str, source: str):
        """
        Processes a guideline chunk through the local Ollama pipeline.
        """
        await self.graph.add_episode(
            name=f"Source: {source}",
            episode_body=text,
            source_description=source,
            reference_time=datetime.now()
        )

    async def search_related_context(self, query: str):
        """Hybrid search using local reranking and graph traversal."""
        return await self.graph.search(query)