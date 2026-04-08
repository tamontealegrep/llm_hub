from app.capabilities.agents.contracts import AgentRunRequest, AgentRunResult
from app.runtime.execution.agent_orchestrator import AgentOrchestrator


class AgentService:
    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        self._orchestrator = orchestrator

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        return await self._orchestrator.run(request)