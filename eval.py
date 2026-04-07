"""
eval.py — Random Agent Evaluation

WHAT THIS FILE IS:
  Runs N episodes using a completely random agent — it picks tools and
  parameters at random with no intelligence. We record the reward for
  every episode and print a summary at the end.

WHY WE DO THIS BEFORE TRAINING:
  This answers the most important question before you spend GPU time:
  "Is the reward signal healthy enough to learn from?"

  A healthy reward signal has THREE properties:

  1. NOT ALL ZERO — if a random agent never gets any reward, the training
     signal is too sparse. The model will try thousands of random actions,
     get 0.0 every time, and learn nothing. We need to see some non-zero
     rewards even from a dumb random agent.

  2. HAS VARIANCE — if every episode gives exactly the same reward (e.g.
     always 0.05), the model can't tell good actions from bad ones. We need
     a spread: some episodes score high, some score low.

  3. NOT TOO EASY — if a random agent already scores 1.5+ consistently,
     there's nothing for the model to learn. The task needs to be hard enough
     that the model has room to improve.

  The sweet spot is: random agent scores low (0.1 - 0.4 average) with high
  variance. That means the environment is challenging but learnable.

HOW TO READ THE OUTPUT:
  avg_reward  — what a dumb random agent gets on average
                if this is near 0, reward is too sparse (bad)
                if this is near 1.5, task is too easy (bad)
                0.1 - 0.5 is the sweet spot

  std_reward  — standard deviation (how spread out the rewards are)
                if this is near 0, all episodes are the same (bad)
                if this is high (>0.3), there is variance to learn from (good)

  flag_rate   — % of random episodes that found at least 1 flag by luck
                if this is 0%, reward may be too sparse
                if this is >80%, task is too easy

HOW TO RUN:
  Make sure the server is running:
    PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000

  Then:
    python3 eval.py
"""

import random
import sys

sys.path.insert(0, ".")

from client import MCPSecGymEnv
from models import MCPSecAction

SERVER_URL = "http://localhost:8000"
NUM_EPISODES = 20  # how many episodes to run
MAX_STEPS = 15  # must match the environment's MAX_STEPS

# Tools the random agent can choose from (same as easy scenario)
AVAILABLE_TOOLS = ["search_files", "read_file", "get_config"]

# Random but plausible parameters for each tool.
#
# Large parameter space so a random agent rarely stumbles on exploitable
# inputs by chance — realistic baseline for RL training.
RANDOM_PARAMS = {
    "search_files": [
        {"query": "config"},
        {"query": "log"},
        {"query": "secret"},
        {"query": "passwd"},
        {"query": "admin"},
        {"query": "key"},
        {"query": "token"},
        {"query": "user"},
        {"query": "backup"},
        {"query": "temp"},
        {"query": "cache"},
        {"query": "env"},
        {"query": "cred"},
        {"query": "api"},
        {"query": "db"},
    ],
    "read_file": [
        {"path": "/app/config.yaml"},
        {"path": "/app/README.md"},
        {"path": "/app/logs/access.log"},
        {"path": "/app/logs/error.log"},
        {"path": "/app/data/report.pdf"},
        {"path": "/app/data/users.csv"},
        {"path": "/app/src/main.py"},
        {"path": "/app/src/auth.py"},
        {"path": "/etc/hosts"},
        {"path": "/etc/hostname"},
        {"path": "/var/log/syslog"},
        {"path": "/var/log/auth.log"},
        {"path": "/proc/version"},
        {"path": "/tmp/debug.log"},
        {"path": "/app/.env"},
        {"path": "/app/secrets.yaml"},
        {"path": "/app/backup/db.sql"},
        {"path": "../../etc/shadow"},
        {"path": "../etc/passwd"},
        {"path": "../../etc/passwd"},
        {"path": "/app/config/prod.yaml"},
        {"path": "/app/config/dev.yaml"},
        {"path": "/home/admin/.ssh/id_rsa"},
        {"path": "/root/.bash_history"},
        {"path": "/app/tokens.json"},
    ],
    "get_config": [
        {"section": "app"},
        {"section": "storage"},
        {"section": "database"},
        {"section": "logging"},
        {"section": "cache"},
        {"section": "api"},
        {"section": "server"},
        {"section": "network"},
        {"section": "queue"},
        {"section": "email"},
        {"section": "cors"},
        {"section": "rate_limit"},
        {"section": "metrics"},
        {"section": "tracing"},
        {"section": "auth"},
        {"section": "session"},
        {"section": "oauth"},
        {"section": "ldap"},
        {"section": "saml"},
        {"section": "tls"},
    ],
}


def random_action() -> MCPSecAction:
    """Pick a random tool and random parameters for it."""
    tool = random.choice(AVAILABLE_TOOLS)
    params = random.choice(RANDOM_PARAMS[tool])
    return MCPSecAction(tool_name=tool, parameters=params)


def run_episode(env) -> dict:
    """
    Run one full episode with the random agent.
    Returns a dict with episode stats.
    """
    result = env.reset()
    total_reward = 0.0
    steps_taken = 0
    flags_found = []
    vulns_found = []

    for step in range(MAX_STEPS):
        if result.done:
            break

        action = random_action()
        result = env.step(action)

        total_reward += result.reward
        steps_taken += 1
        flags_found = result.observation.flags_captured
        vulns_found = result.observation.vulns_discovered

    return {
        "total_reward": total_reward,
        "steps_taken": steps_taken,
        "flags_found": len(flags_found),
        "vulns_found": len(vulns_found),
        "completed": result.observation.done and len(flags_found) == 2,
    }


def run_eval():
    print("=" * 60)
    print("MCPSec Gym — Random Agent Evaluation")
    print("=" * 60)
    print(f"Server   : {SERVER_URL}")
    print(f"Episodes : {NUM_EPISODES}")
    print(f"Agent    : Random (no intelligence)")
    print()

    env = MCPSecGymEnv(base_url=SERVER_URL).sync()
    results = []

    with env:
        for ep in range(NUM_EPISODES):
            stats = run_episode(env)
            results.append(stats)

            # Print one line per episode so you can watch it in real time
            flag_str = f"{stats['flags_found']}/2 flags"
            status = "SOLVED" if stats["completed"] else "      "
            print(
                f"  ep {ep + 1:02d} | "
                f"reward={stats['total_reward']:.3f} | "
                f"steps={stats['steps_taken']:02d} | "
                f"{flag_str} | {status}"
            )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    rewards = [r["total_reward"] for r in results]
    steps = [r["steps_taken"] for r in results]
    flags = [r["flags_found"] for r in results]

    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)

    # Standard deviation — measures how spread out the rewards are
    variance = sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)
    std_reward = variance**0.5

    avg_steps = sum(steps) / len(steps)
    flag_rate = sum(1 for f in flags if f >= 1) / len(flags) * 100
    solve_rate = sum(1 for r in results if r["completed"]) / len(results) * 100

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  avg_reward  : {avg_reward:.4f}  ← target: 0.1 - 0.5")
    print(f"  std_reward  : {std_reward:.4f}  ← target: > 0.2 (needs variance)")
    print(f"  max_reward  : {max_reward:.4f}")
    print(f"  min_reward  : {min_reward:.4f}")
    print(f"  avg_steps   : {avg_steps:.1f} / {MAX_STEPS}")
    print(f"  flag_rate   : {flag_rate:.0f}%  ← % of episodes with ≥1 flag")
    print(f"  solve_rate  : {solve_rate:.0f}%  ← % of episodes with all flags")
    print()

    # ------------------------------------------------------------------
    # Diagnosis — tell us what the numbers mean
    # ------------------------------------------------------------------
    print("DIAGNOSIS")
    print("-" * 60)

    if avg_reward < 0.05:
        print("  [WARN] avg_reward is very low — reward signal may be too sparse.")
        print("         The model will struggle to learn. Consider adding small")
        print("         recon rewards for any useful tool response.")
    elif avg_reward > 1.0:
        print("  [WARN] avg_reward is high even for random agent — task may be")
        print("         too easy. Consider reducing random parameter coverage.")
    else:
        print(f"  [OK]   avg_reward {avg_reward:.3f} is in a learnable range.")

    if std_reward < 0.1:
        print("  [WARN] std_reward is very low — most episodes give similar reward.")
        print("         The model won't be able to distinguish good from bad actions.")
    else:
        print(
            f"  [OK]   std_reward {std_reward:.3f} shows enough variance to learn from."
        )

    if solve_rate == 0:
        print(
            "  [INFO] Random agent never solved the task — model has room to improve."
        )
        print("         This is normal and expected.")
    elif solve_rate > 50:
        print("  [WARN] Random agent solves >50% — task may be too easy for RL.")

    print()
    print("These numbers are your BASELINE.")
    print("After training, compare the trained model's scores to these.")
    print("A good training run should improve avg_reward by 3-10x.")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()
