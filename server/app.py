# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the MCPSec Gym — Multi-Task Environment.

This server hosts all 3 MCPSec scenarios under one endpoint. The active
task is selected by passing task="easy"|"medium"|"hard" to reset().

Endpoints:
    - POST /reset  : Reset the environment (body: {"task": "easy"})
    - POST /step   : Execute an action
    - GET  /state  : Get current environment state
    - GET  /schema : Get action/observation schemas
    - WS   /ws     : WebSocket endpoint for persistent sessions

Usage (development):
    PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000

Usage (production / Docker):
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with:\n    uv sync\n"
    ) from e

try:
    from ..models import MCPSecAction, MCPSecObservation
    from .mcpsec_gym_environment import McpsecGymEnvironment
    from .medium_environment import MediumEnvironment
    from .hard_environment import HardEnvironment
except ImportError:
    from models import MCPSecAction, MCPSecObservation
    from server.mcpsec_gym_environment import McpsecGymEnvironment
    from server.medium_environment import MediumEnvironment
    from server.hard_environment import HardEnvironment

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State


class MCPSecMultiTaskEnvironment(Environment):
    """
    Dispatcher environment that routes to the appropriate MCPSec scenario
    based on the 'task' kwarg passed to reset().

    Supported tasks:
      "easy"   → config_leak   (2 flags, 3 tools, no defenses)
      "medium" → chain_reaction (3 flags, 5 tools, rate limiter)
      "hard"   → fortress_breach (3 flags, 7 tools, full defense stack)

    Default task when not specified: "easy"
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        # Start with easy as the default
        self._delegate: Environment = McpsecGymEnvironment()
        self._current_task: str = "easy"

    def _get_delegate(self, task: str) -> Environment:
        """Return the correct delegate environment for the given task string."""
        if task == "easy":
            return McpsecGymEnvironment()
        elif task == "medium":
            return MediumEnvironment()
        elif task == "hard":
            return HardEnvironment()
        else:
            # Unknown task — fall back to easy
            return McpsecGymEnvironment()

    def reset(self, seed=None, **kwargs) -> tuple:
        """
        Reset the environment, optionally switching the active task.

        kwargs:
            task (str): "easy" | "medium" | "hard"  (default: "easy")
            seed (int | None): RNG seed for reproducibility (default: None = random)

        Returns:
            (MCPSecObservation, {"variant_id": str})
        """
        task = kwargs.pop("task", "easy").lower()
        self._delegate = self._get_delegate(task)
        self._current_task = task

        result = self._delegate.reset(seed=seed, **kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        self._last_info = info
        return obs, info

    async def reset_async(self, seed=None, **kwargs) -> MCPSecObservation:
        """
        Async reset used by the OpenEnv HTTP/WebSocket server.

        Returns only the observation (not the info dict) to satisfy the
        openenv framework's serialize_observation() contract. The info dict
        is stored in self._last_info and accessible via get_info().
        """
        obs, _info = self.reset(seed=seed, **kwargs)
        return obs

    def get_info(self) -> dict:
        """Return the info dict from the last reset() call."""
        return getattr(self, "_last_info", {})

    def step(self, action: MCPSecAction, **kwargs) -> MCPSecObservation:  # type: ignore[override]
        """Forward step to the active delegate environment."""
        return self._delegate.step(action, **kwargs)

    @property
    def state(self) -> State:
        return self._delegate.state


# Wire up all OpenEnv endpoints to the multi-task dispatcher.
# max_concurrent_envs=10 supports parallel GRPO rollouts.
app = create_app(
    MCPSecMultiTaskEnvironment,
    MCPSecAction,
    MCPSecObservation,
    env_name="mcpsec_gym",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
