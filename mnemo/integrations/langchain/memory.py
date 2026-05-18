"""LangChain integration for memori.

Usage:
    from mnemo.integrations.langchain import MemoriMemory
    from langchain.agents import create_react_agent

    memory = MemoriMemory(client=memori_client)
    agent = create_react_agent(llm, tools, memory=memory)
"""

from __future__ import annotations

from typing import Any, Optional


class MemoriMemory:
    """LangChain-compatible memory class backed by memori.

    Implements the BaseMemory interface expected by LangChain agents.
    """

    def __init__(self, client, session_id: Optional[str] = None):
        self.client = client
        self.session_id = session_id

    @property
    def memory_variables(self) -> list[str]:
        return ["chat_history", "relevant_memories"]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load memory context for the agent."""
        query = inputs.get("input", "")
        memories = []
        if query:
            results = self.client.recall(query, top_k=5)
            memories = [r["memory"]["content"] for r in results]

        return {
            "chat_history": [],
            "relevant_memories": "\n".join(memories) if memories else "",
        }

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Save the interaction to memory."""
        user_input = inputs.get("input", "")
        agent_output = outputs.get("output", "")

        if user_input:
            self.client.store(
                f"User: {user_input}",
                memory_type="episodic",
                session_id=self.session_id,
            )
        if agent_output:
            self.client.store(
                f"Agent: {agent_output}",
                memory_type="episodic",
                session_id=self.session_id,
            )

    def clear(self) -> None:
        """Clear memory (stub — implement selective clear)."""
        pass
