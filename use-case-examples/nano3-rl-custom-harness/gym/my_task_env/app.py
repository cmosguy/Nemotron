# SPDX-License-Identifier: Apache-2.0
"""NeMo Gym Resources Server for your tasks: tools (optional) + the reward function.

Copy to `resources_servers/my_task_env/` in your fork of NeMo Gym. Structure follows the
Gym single-step environment tutorial (SimpleResourcesServer + a separate verifier mixin so
rewards are unit-testable and saved rollouts can be re-verified).

The reward here mirrors Nano3's workplace_assistant environment: compare the function calls
the agent made with the expected ones from the task row, and optionally check the final
answer text. Replace `verify()` with your own definition of "the agent did the job",
ideally by checking the resulting state of your system, not the exact wording.
"""


import json
from typing import Any, ClassVar, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

# Import paths as of the nemo-gym 0.7.0 release (Gym main moved these to nemo_gym.server_runtime / nemo_gym.utils).
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint


class MyTaskEnvConfig(BaseResourcesServerConfig):
    # Reward depends only on the request (row + transcript), so saved rollouts can be re-scored.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


# Fields from the task row (written by convert_trajectories.py rl) arrive on the verify request.
class MyTaskVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    expected_calls: Optional[list[dict[str, Any]]] = None
    expected_answer_contains: Optional[list[str]] = None


class MyTaskVerifyResponse(BaseVerifyResponse):
    # Keep row fields such as task_id in the rollout output, for per-task pass-rate profiling.
    model_config = ConfigDict(extra="allow")


def _args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"__unparseable__": raw}


class MyTaskVerifier:
    async def verify(self, body: MyTaskVerifyRequest) -> MyTaskVerifyResponse:
        output = body.response.output
        calls = [(o.name, _args(o.arguments)) for o in output if o.type == "function_call"]
        final_text = " ".join(
            c.text for o in output if o.type == "message" for c in (o.content or []) if getattr(c, "text", None)
        )

        checks: list[bool] = []
        if body.expected_calls:
            # Every expected call must appear (order-insensitive); extra read-only calls are allowed.
            for exp in body.expected_calls:
                checks.append(any(n == exp["name"] and a == _args(exp["arguments"]) for n, a in calls))
            # Penalize state-changing calls that were not expected (e.g. closing the wrong ticket).
            expected_writes = {(e["name"], json.dumps(_args(e["arguments"]), sort_keys=True)) for e in body.expected_calls}
            for n, a in calls:
                if n.startswith(("update_", "delete_", "create_", "send_")):
                    checks.append((n, json.dumps(a, sort_keys=True)) in expected_writes)
        if body.expected_answer_contains:
            checks.extend(s.lower() in final_text.lower() for s in body.expected_answer_contains)

        reward = 1.0 if checks and all(checks) else 0.0
        return MyTaskVerifyResponse(**body.model_dump(), reward=reward)


# ---- Optional tools, if your harness lets Gym host them (stateless stubs here) ----------
class SearchTicketsRequest(BaseModel):
    query: str
    status: Optional[str] = None


class UpdateTicketRequest(BaseModel):
    ticket_id: str
    status: Optional[str] = None
    assignee: Optional[str] = None


class MyTaskEnv(MyTaskVerifier, SimpleResourcesServer):
    config: MyTaskEnvConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/search_tickets")(self.search_tickets)
        app.post("/update_ticket")(self.update_ticket)
        return app

    async def search_tickets(self, body: SearchTicketsRequest) -> dict[str, Any]:
        # Replace with a call to a sandboxed copy of your backend, keyed by the session cookie.
        return {"results": []}

    async def update_ticket(self, body: UpdateTicketRequest) -> dict[str, Any]:
        return {"ok": True}


if __name__ == "__main__":
    MyTaskEnv.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = MyTaskEnv.run_webserver()  # noqa: F401
