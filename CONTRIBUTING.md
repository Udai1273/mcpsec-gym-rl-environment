# Contributing to MCPSec Gym

So you want to join the crew. Good.

MCPSec Gym is an open RL environment for training agents to find security vulnerabilities in MCP servers. We built three missions. We think there should be more. Maybe you're the one to build the next one.

Here's how to get involved — whether you're fixing a typo, adding a variant, or designing a whole new scenario from scratch.

---

## The Quick Version

1. Fork the repo
2. Create a branch (`git checkout -b my-change`)
3. Make your changes
4. Run the tests (`PYTHONPATH=. uv run pytest test_easy.py test_medium.py test_hard.py -q`)
5. Commit with a clear message
6. Open a PR

That's it for small stuff — typos, doc improvements, bug fixes. For anything bigger, keep reading.

---

## Adding a New Variant

Each mission (easy, medium, hard) has 4 hand-authored variants. A variant is a specific configuration of the scenario — which config section leaks, which file is traversable, which user ID has the IDOR.

Want to add a 5th variant? Here's the pattern:

1. Open the relevant environment file (`server/mcpsec_gym_environment.py`, `medium_environment.py`, or `hard_environment.py`)
2. Find the `_VARIANTS` list near the top of the class
3. Add your variant dict following the exact same schema as the others
4. Update the variant selection: `self._rng.randint(0, len(self._VARIANTS) - 1)`
5. Write tests — at minimum, one deterministic test with a fixed seed that exercises your variant

**The rules:**
- Flags must be generated from `self._rng`, not hardcoded
- Your variant must work with the existing tools — don't add new tools to an existing mission
- The reward budget must still sum to 1.0
- Run `PYTHONPATH=. uv run pytest -q` and see green before you PR

---

## Building a New Mission

This is the big one. A new difficulty level, a new scenario, a whole new attack surface.

Here's what a mission needs:

### 1. An environment file

Create `server/your_environment.py`. Follow the pattern in the existing files:

```python
class YourEnvironment:
    def __init__(self):
        self._rng = None           # seeded per-episode
        self._step_count = 0
        self._max_steps = ...
        self._flags_captured = []
        self._vulns_discovered = []
        # ... your state

    def reset(self, seed=None):
        """Pick a variant, generate flags, return initial observation."""
        ...

    def step(self, action):
        """Process a tool call, return observation + reward."""
        ...
```

**Non-negotiable requirements:**
- Instance-level RNG: `self._rng = random.Random(seed)` — never `import random` at module level
- Flag format: `FLAG{<8 hex chars>}` generated from `self._rng`
- All state on `self._` — no globals, no class variables
- Reward clamped to [-1.0, 1.0] in `_make_obs`
- Recon bonus fires once per episode max (use a `_recon_bonus_given` guard)

### 2. Register it in the dispatcher

Add your task to `server/app.py`:

```python
# In the reset endpoint
if task == "your_task":
    env = YourEnvironment()
```

And export it from `server/__init__.py`.

### 3. Write tests

Create `test_your_task.py`. Minimum coverage:

- Deterministic variant selection with fixed seeds
- Every flag is findable
- Every vulnerability type is discoverable
- Reward budget sums to 1.0
- Invalid tool calls return -0.05 penalty
- Step budget enforcement (episode ends at max steps)
- If you have defenses (rate limiter, honeypot, lockout), test those too

Aim for 20+ tests. We have 69 across 3 missions. Keep the bar high.

### 4. Update the scripts

- `eval.py` — add your task to the `--task` choices
- `train.py` — add a GRPO config section for your task
- `inference.py` — add a strategy (or verify the LLM fallback works)

### 5. Update openenv.yaml

Add your task to the `tasks` section with a description, tool list, and evaluation baselines.

---

## Mission Ideas (Free to Take)

Here are scenarios we've talked about but haven't built yet. If one of these sounds interesting, grab it:

- **SSRF Chain** — An MCP server that proxies requests. The agent needs to find an internal endpoint that's not supposed to be reachable from outside, then chain it with a parameter injection to read internal metadata.
- **Auth Bypass** — A server with role-based access control. Some endpoints check permissions, others don't. The agent needs to map the permission model and find the unprotected admin endpoint.
- **Supply Chain** — A package registry MCP server. One of the packages has a dependency that's been tampered with. The agent needs to trace the dependency tree and find the malicious payload.
- **Log Poisoning** — An MCP server that writes user input to log files. The agent needs to inject a payload into the logs, then trigger the log parser to execute it.

Each of these maps to real CVE patterns. That's the whole point — train on realistic attack chains, not made-up puzzles.

---

## Code Style

Nothing exotic:

- Python 3.10+
- Type hints where they help readability
- Docstrings on public methods
- No module-level state (this one matters — it breaks concurrent episodes)
- `uv` for dependency management
- Tests with `pytest`

We don't enforce a formatter. Just make it readable.

---

## Running the Full Suite

```bash
# Start the server
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000 &

# Run all tests
PYTHONPATH=. uv run pytest test_easy.py test_medium.py test_hard.py -q

# Validate the OpenEnv contract
openenv validate

# Run the random agent across all missions
PYTHONPATH=. uv run python eval.py --task all
```

All tests must pass. `openenv validate` must return 6/6. These are not suggestions.

---

## Commit Messages

We don't enforce a format, but we appreciate clarity:

```
fix: clamp reward to [-1.0, 1.0] server-side
feat: add 5th variant to easy mode (XSS in error page)
test: add lockout escalation tests for hard mode
docs: update reward table in README
```

One logical change per commit. We review diffs, not novels.

---

## Questions?

Open an issue. Or just start building — a PR with working tests speaks louder than a proposal.

We built MCPSec Gym because we wanted to see AI agents learn to think like pentesters. If that sounds like a problem worth working on, you're in the right place.
