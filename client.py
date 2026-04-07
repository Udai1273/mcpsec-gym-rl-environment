"""
MCPSec Gym Client

This is the Python client that your training code uses to talk to the
environment server over WebSocket.

WHY A CLIENT EXISTS:
  The environment runs as a separate HTTP server (in Docker or locally).
  Your training script doesn't import the environment directly — it talks to
  the server through this client. This is how OpenEnv works: server and
  client are separate. The client handles all the networking so your training
  code only calls clean Python methods: reset(), step(), state().

HOW IT WORKS:
  1. client = MCPSecGymEnv(base_url="http://localhost:8000")
  2. client.reset()  →  sends a WebSocket message to /ws  →  returns StepResult
  3. client.step(action)  →  sends action as JSON  →  returns StepResult

THE THREE METHODS YOU MUST IMPLEMENT:
  _step_payload()  — converts your Action object INTO json to send to server
  _parse_result()  — converts json FROM the server INTO your Observation object
  _parse_state()   — converts json FROM /state endpoint INTO a State object

Everything else (WebSocket connection, retries, timeouts) is handled by the
base class EnvClient.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import MCPSecAction, MCPSecObservation
except ImportError:
    from models import MCPSecAction, MCPSecObservation


class MCPSecGymEnv(EnvClient[MCPSecAction, MCPSecObservation, State]):
    """
    WebSocket client for the MCPSec Gym environment.

    Usage (sync wrapper — easiest for scripts and tests):
        env = MCPSecGymEnv(base_url="http://localhost:8000").sync()
        with env:
            result = env.reset()
            result = env.step(MCPSecAction(tool_name="get_config", parameters={"section": "auth"}))
            print(result.observation.flags_captured)

    Usage (async — needed for GRPO rollout functions):
        async with MCPSecGymEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(MCPSecAction(tool_name="get_config", parameters={}))
    """

    def _step_payload(self, action: MCPSecAction) -> Dict[str, Any]:
        """
        Convert an MCPSecAction into the JSON dict the server expects.

        The server's /step endpoint expects:
            {"action": {"tool_name": "...", "parameters": {...}}}

        We return just the inner dict — the base class wraps it in {"action": ...}
        """
        return {
            "tool_name": action.tool_name,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MCPSecObservation]:
        """
        Convert the JSON response from the server into a StepResult.

        The server returns:
            {
                "observation": {
                    "tool_response": "...",
                    "available_tools": [...],
                    "flags_captured": [...],
                    "vulns_discovered": [...],
                    "alert_level": 0,
                    "steps_remaining": 14,
                    "step_count": 1,
                    "done": false,
                    "reward": 0.75
                },
                "reward": 0.75,
                "done": false
            }

        We unpack the "observation" dict into our MCPSecObservation object.
        """
        obs_data = payload.get("observation", {})

        observation = MCPSecObservation(
            tool_response=obs_data.get("tool_response", ""),
            available_tools=obs_data.get("available_tools", []),
            flags_captured=obs_data.get("flags_captured", []),
            vulns_discovered=obs_data.get("vulns_discovered", []),
            alert_level=obs_data.get("alert_level", 0),
            steps_remaining=obs_data.get("steps_remaining", 0),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Convert the JSON response from the /state endpoint into a State object.

        State just tracks the episode ID and how many steps have been taken.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
