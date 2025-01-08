import os
from typing import Any, Dict, List

from mem0 import MemoryClient,Memory
from Crewai.memory.storage.interface import Storage
from env import CACHE_DIR

class Mem0Storage(Storage):
    """
    Extends Storage to handle embedding and searching across entities using Mem0.
    """

    def __init__(self, type, crew=None):
        super().__init__()

        if type not in ["user", "short_term", "long_term", "entities"]:
            raise ValueError("Invalid type for Mem0Storage. Must be 'user' or 'agent'.")

        self.memory_type = type
        self.crew = crew
        self.memory_config = crew.memory_config

        # User ID is required for user memory type "user" since it's used as a unique identifier for the user.
        user_id = self._get_user_id()
        if type == "user" and not user_id:
            raise ValueError("User ID is required for user memory type")

        # API key in memory config overrides the environment variable
        mem0_api_key = self.memory_config.get("config", {}).get("api_key") or os.getenv(
            "MEM0_API_KEY"
        )
        config = {
            "llm": {
                "provider": "groq",
                "config": {
                    "model": "mixtral-8x7b-32768",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "multi-qa-MiniLM-L6-cos-v1"
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "test",
                    "path": os.path.join(CACHE_DIR,"memo",'history.db'),
                }
            }
        }

        if mem0_api_key:
            self.memory = MemoryClient(api_key=mem0_api_key)
        else:
            self.memory =Memory.from_config(config)
    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        user_id = self._get_user_id()
        agent_name = self._get_agent_name()
        if self.memory_type == "user":
            self.memory.add(value, user_id=user_id, metadata={**metadata})
        elif self.memory_type == "short_term":
            agent_name = self._get_agent_name()
            self.memory.add(
                value, agent_id=agent_name, metadata={"type": "short_term", **metadata}
            )
        elif self.memory_type == "long_term":
            agent_name = self._get_agent_name()
            self.memory.add(
                value,
                agent_id=agent_name,
                infer=False,
                metadata={"type": "long_term", **metadata},
            )
        elif self.memory_type == "entities":
            entity_name = None
            self.memory.add(
                value, user_id=entity_name, metadata={"type": "entity", **metadata}
            )

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        params = {"query": query, "limit": limit}
        if self.memory_type == "user":
            user_id = self._get_user_id()
            params["user_id"] = user_id
        elif self.memory_type == "short_term":
            agent_name = self._get_agent_name()
            params["agent_id"] = agent_name
        elif self.memory_type == "long_term":
            agent_name = self._get_agent_name()
            params["agent_id"] = agent_name
        elif self.memory_type == "entities":
            agent_name = self._get_agent_name()
            params["agent_id"] = agent_name

        # Discard the filters for now since we create the filters
        # automatically when the crew is created.
        results = self.memory.search(**params)
        return [r for r in results if r["score"] >= score_threshold]

    def _get_user_id(self):
        if self.memory_type == "user":
            if hasattr(self, "memory_config") and self.memory_config is not None:
                return self.memory_config.get("config", {}).get("user_id")
            else:
                return None
        return None

    def _get_agent_name(self):
        agents = self.crew.agents if self.crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        return agents
