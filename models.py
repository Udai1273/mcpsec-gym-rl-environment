"""
Data models for MCPSec Gym.

Three classes:
  - MCPSecAction    : what the agent SENDS each step (which tool + what args)
  - MCPSecObservation : what the agent RECEIVES back (tool output + game state)

Both inherit from OpenEnv base classes so the HTTP server knows how to
serialize/deserialize them automatically.
"""

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MCPSecAction(Action):
    """
    One action = one MCP tool call.

    The agent picks a tool by name and passes a dictionary of arguments.

    Example:
        MCPSecAction(tool_name="get_config", parameters={"section": "auth"})
    """

    # Which tool to call. The environment will reject unknown tool names.
    tool_name: str = Field(..., description="Name of the MCP tool to call")

    # Free-form key-value arguments for the tool.
    # Dict[str, Any] means keys are strings, values can be anything (str, int, etc.)
    # default_factory=dict means "start with an empty dict if nothing is provided"
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool",
    )


class MCPSecObservation(Observation):
    """
    What the agent sees after each step.

    Inherits from Observation which already has:
      - done: bool        (is the episode over?)
      - reward: float     (how much reward this step earned)
      - metadata: dict    (any extra info)

    We add MCPSec-specific fields below.
    """

    # The raw text response from the tool the agent just called.
    # This is what the agent reads to decide its next action.
    tool_response: str = Field(
        default="",
        description="Output returned by the MCP tool",
    )

    # Which tools the agent is allowed to call right now.
    # This changes per scenario (easy has 3, hard has 8).
    available_tools: List[str] = Field(
        default_factory=list,
        description="Tool names the agent can use this episode",
    )

    # Flags the agent has captured so far in this episode.
    # A flag looks like: FLAG{some_secret_value}
    # Capturing all flags ends the episode successfully.
    flags_captured: List[str] = Field(
        default_factory=list,
        description="Flags found so far (strings starting with FLAG{)",
    )

    # Vulnerability types the agent has triggered so far.
    # Example values: "info_leak", "path_traversal", "sql_injection"
    # Used in the grader to compute the vuln-discovery score component.
    vulns_discovered: List[str] = Field(
        default_factory=list,
        description="Vulnerability types triggered so far",
    )

    # How suspicious the defense system thinks the agent is.
    # 0 = completely undetected. 5+ = locked out.
    # Easy scenario: always 0 (no defenses).
    alert_level: int = Field(
        default=0,
        description="Defense alert level (0=clean, 5+=lockout)",
    )

    # How many steps are left before the episode times out.
    # Starts at max_steps and counts down each step.
    steps_remaining: int = Field(
        default=0,
        description="Steps left before episode times out",
    )

    # How many steps have been taken so far (starts at 0).
    step_count: int = Field(
        default=0,
        description="Number of steps taken so far",
    )
