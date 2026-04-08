from app.capabilities.agents.contracts import (
    AgentRunRequest,
    AgentRunResult
)


class AgentOrchestrator:
    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        raise NotImplementedError(
            "AgentOrchestrator aún no está implementado. "
            "Se añadirá cuando integres LangGraph o un runtime de agentes."
        )