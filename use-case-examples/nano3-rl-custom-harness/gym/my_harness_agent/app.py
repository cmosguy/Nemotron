# SPDX-License-Identifier: Apache-2.0
"""NeMo Gym Agent Server adapter for a proprietary agent harness (skeleton).

Copy this directory to `responses_api_agents/my_harness_agent/` in your (private) fork of
NeMo Gym. It follows the same pattern as Gym's built-in `claude_code_agent` adapter:

    /run  ->  seed_session on the Resources Server
          ->  run YOUR harness in-process, pointed at the Gym model server
              (URL carries /ng-rollout/<id>/training-token-capture/ so every call is recorded)
          ->  convert the harness transcript to a NeMoGymResponse
          ->  /verify on the Resources Server  ->  reward

Requirements for RL training (not just evaluation):
  1. EVERY model call the harness makes goes to `model_base_url` below. A call that goes
     anywhere else (a hosted API, a side model, a summarizer) is not captured and Gym masks
     the rollout (`mask_sample: true`).
  2. The conversation only grows. Each call's prompt must start with the previous call's
     prompt + sampled output. Harnesses that compact, summarize or rewrite history produce
     multiple chains; Gym trains on one verified chain per rollout.
  3. Token capture needs NeMo Gym >= 0.6 (see README: version notes).

Checked against nemo-gym 0.7.0 (PyPI): imports, endpoint signatures, and the transcript ->
response -> verify path. The harness call itself is a stub.
"""


import json
import logging
from asyncio import Semaphore
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict

# Import paths as of the nemo-gym 0.7.0 release. Gym's main branch has since moved some of these
# (nemo_gym.core, nemo_gym.server_runtime, nemo_gym.client, nemo_gym.config); match your version.
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status

LOG = logging.getLogger(__name__)


class MyHarnessAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef
    max_turns: int = 30
    concurrency: int = 64  # per worker; rollouts in flight at once


class MyHarnessRunRequest(BaseRunRequest):
    # extra="allow" keeps your verifier fields (expected_calls, ...) on the row so they reach /verify.
    model_config = ConfigDict(extra="allow")


class MyHarnessVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class MyHarnessAgent(SimpleResponsesAPIAgent):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    config: MyHarnessAgentConfig
    sem: Semaphore = None

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    # ------------------------------------------------------------------------------
    # The one function you implement: run your harness against an OpenAI-compatible URL.
    # ------------------------------------------------------------------------------
    async def run_my_harness(
        self,
        *,
        model_base_url: str,
        params: NeMoGymResponseCreateParamsNonStreaming,
        resources_cookies: Any,
    ) -> dict[str, Any]:
        """Run one episode of your harness and return its transcript.

        `model_base_url` ends in /v1 and speaks the OpenAI API (chat completions and responses),
        served by vLLM inside NeMo RL. Configure your harness's client with it, for example
        `OpenAI(base_url=model_base_url, api_key="EMPTY")`, and send every call there.

        Tools: either your harness executes them itself (fine for training), or it forwards
        them to the Resources Server so the verifier can inspect state:

            r = await self.server_client.post(server_name=self.config.resources_server.name,
                                              url_path=f"/{tool_name}", json=args, cookies=resources_cookies)

        Return the transcript in OpenAI chat format plus the id of the last model response:
            {"messages": [...assistant/tool turns produced in this episode...],
             "terminal_response_id": "<id of the last served model response, if your client exposes it>"}
        """
        raise NotImplementedError("Call your proprietary harness here.")

    # ------------------------------------------------------------------------------
    # Transcript (chat format) -> Responses API output items the verifier reads.
    # Under token capture, Gym replaces these with token-bearing items after the run.
    # ------------------------------------------------------------------------------
    def transcript_to_response(
        self, params: NeMoGymResponseCreateParamsNonStreaming, transcript: dict[str, Any]
    ) -> NeMoGymResponse:
        output: list[dict[str, Any]] = []
        for m in transcript.get("messages", []):
            if m["role"] == "assistant":
                if m.get("content"):
                    output.append({
                        "id": f"msg_{uuid4().hex}", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": m["content"], "annotations": []}],
                    })
                for c in m.get("tool_calls") or []:
                    output.append({
                        "id": f"fc_{uuid4().hex}", "type": "function_call", "status": "completed",
                        "call_id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"],
                    })
            elif m["role"] == "tool":
                output.append({"type": "function_call_output", "call_id": m["tool_call_id"],
                               "output": m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])})
        # Naming the response after the last served model call lets Gym's terminal attribution
        # find the chain the verifier scored (the built-in Claude Code adapter does the same).
        terminal_id = transcript.get("terminal_response_id")
        response_id = terminal_id if (terminal_id and self._token_id_capture_enabled()) else f"resp_{uuid4().hex}"
        return NeMoGymResponse.model_validate({
            "id": response_id, "created_at": int(time()), "model": params.get("model") or "policy",
            "object": "response", "output": output, "tool_choice": params.get("tool_choice", "auto"),
            "tools": params.get("tools", []), "parallel_tool_calls": params.get("parallel_tool_calls", False),
        })

    # ------------------------------------------------------------------------------
    # Gym endpoints
    # ------------------------------------------------------------------------------
    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """Single episode without seeding/verification (used by some evaluation paths)."""
        rollout_id: Optional[str] = request.path_params.get("rollout_id")
        base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        transcript = await self.run_my_harness(model_base_url=base_url, params=body, resources_cookies=request.cookies)
        return self.transcript_to_response(body, transcript)

    async def run(self, request: Request, body: MyHarnessRunRequest) -> MyHarnessVerifyResponse:
        async with self.sem:
            seed = await self.server_client.post(
                server_name=self.config.resources_server.name, url_path="/seed_session",
                json=body.model_dump(), cookies=request.cookies,
            )
            await raise_for_status(seed)
            cookies = seed.cookies

            # .../ng-rollout/<rollout_id>/training-token-capture/v1 when capture is on for this agent.
            rollout_id = self.rollout_id_from_run(body)
            base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)

            transcript = await self.run_my_harness(
                model_base_url=base_url, params=body.responses_create_params, resources_cookies=cookies
            )
            agent_resp = self.transcript_to_response(body.responses_create_params, transcript)

            verify = await self.server_client.post(
                server_name=self.config.resources_server.name, url_path="/verify",
                json=body.model_dump() | {"response": agent_resp.model_dump(mode="json")}, cookies=cookies,
            )
            await raise_for_status(verify)
            return MyHarnessVerifyResponse.model_validate(await get_response_json(verify))


if __name__ == "__main__":
    MyHarnessAgent.run_webserver()
