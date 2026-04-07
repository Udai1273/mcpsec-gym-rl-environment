"""
eval.py — Random Agent Evaluation (All Tasks)

Runs N episodes using a completely random agent across easy, medium, and hard
scenarios. Records reward per episode and prints a summary.

WHY WE DO THIS BEFORE TRAINING:
  This answers the most important question before you spend GPU time:
  "Is the reward signal healthy enough to learn from?"

  A healthy reward signal has THREE properties:

  1. NOT ALL ZERO — if a random agent never gets any reward, the training
     signal is too sparse. The model will try thousands of random actions,
     get 0.0 every time, and learn nothing.

  2. HAS VARIANCE — if every episode gives exactly the same reward (e.g.
     always 0.05), the model can't tell good actions from bad ones.

  3. NOT TOO EASY — if a random agent already scores 1.0 consistently,
     there's nothing for the model to learn.

HOW TO RUN:
  Make sure the server is running:
    PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000

  Then:
    python3 eval.py              # runs all 3 tasks
    python3 eval.py --task easy  # runs only easy
    python3 eval.py --task hard --episodes 50
"""

import argparse
import random
import sys

sys.path.insert(0, ".")

from client import MCPSecGymEnv
from models import MCPSecAction

SERVER_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Task configurations — tools and params per difficulty
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "easy": {
        "tools": ["search_files", "read_file", "get_config"],
        "max_steps": 15,
        "total_flags": 2,
    },
    "medium": {
        "tools": ["search_files", "read_file", "get_config", "query_db", "list_users"],
        "max_steps": 20,
        "total_flags": 3,
    },
    "hard": {
        "tools": [
            "search_files",
            "read_file",
            "get_config",
            "query_db",
            "list_users",
            "decode_token",
            "check_endpoint",
        ],
        "max_steps": 25,
        "total_flags": 3,
    },
}

# Random but plausible parameters for each tool
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
        {"path": "/etc/hosts"},
        {"path": "/var/log/syslog"},
        {"path": "/app/.env"},
        {"path": "/app/secrets.yaml"},
        {"path": "../../etc/shadow"},
        {"path": "../etc/passwd"},
        {"path": "../../etc/passwd"},
        {"path": "/app/honeypot.txt"},
        {"path": "/home/admin/.ssh/id_rsa"},
        {"path": "/root/.bash_history"},
    ],
    "get_config": [
        {"section": "app"},
        {"section": "storage"},
        {"section": "database"},
        {"section": "logging"},
        {"section": "cache"},
        {"section": "auth"},
        {"section": "secrets"},
        {"section": "admin_keys"},
        {"section": "internal_tokens"},
        {"section": "service_accounts"},
        {"section": "api_secrets"},
        {"section": "db_root"},
        {"section": "jwt_keys"},
        {"section": "infra_creds"},
        {"section": "vault_tokens"},
    ],
    "query_db": [
        {"table": "users"},
        {"table": "sessions"},
        {"table": "audit_log"},
        {"table": "credentials"},
        {"table": "products"},
        {"table": "users", "filter": "' OR '1'='1"},
        {"table": "sessions", "filter": "id=1"},
        {"table": "credentials", "filter": "' OR '1'='1"},
    ],
    "list_users": [
        {"role": "user"},
        {"role": "admin"},
        {"role": "devops"},
        {"role": "sysadmin"},
        {"role": "superuser"},
        {"role": "superadmin"},
        {},
    ],
    "decode_token": [
        {"token": "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiYWRtaW4ifQ."},
        {"token": "eyJhbGciOiJIUzUxMiJ9.eyJ1c2VyIjoiYWRtaW4ifQ."},
        {"token": "eyJhbGciOiJub25lIn0.eyJ1c2VyIjoiYWRtaW4ifQ."},
        {"token": "invalid_token"},
    ],
    "check_endpoint": [
        {"path": "/internal/admin", "secret": "test"},
        {"path": "/internal/ops", "secret": "test"},
        {"path": "/internal/deploy", "secret": "test"},
        {"path": "/internal/secrets", "secret": "test"},
        {"path": "/health"},
    ],
}


def random_action(tools: list[str]) -> MCPSecAction:
    """Pick a random tool and random parameters for it."""
    tool = random.choice(tools)
    params = random.choice(RANDOM_PARAMS[tool])
    return MCPSecAction(tool_name=tool, parameters=params)


def run_episode(env, task: str) -> dict:
    """Run one full episode with the random agent."""
    cfg = TASK_CONFIG[task]
    result = env.reset(task=task)
    total_reward = 0.0
    steps_taken = 0
    flags_found = []
    vulns_found = []

    for step in range(cfg["max_steps"]):
        if result.done:
            break

        action = random_action(cfg["tools"])
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
        "completed": result.observation.done and len(flags_found) == cfg["total_flags"],
    }


def run_eval(task: str, num_episodes: int):
    cfg = TASK_CONFIG[task]
    print(f"\n{'=' * 60}")
    print(f"MCPSec Gym — Random Agent Evaluation [{task.upper()}]")
    print(f"{'=' * 60}")
    print(f"Server   : {SERVER_URL}")
    print(f"Task     : {task}")
    print(f"Episodes : {num_episodes}")
    print(f"Tools    : {', '.join(cfg['tools'])}")
    print(f"Max steps: {cfg['max_steps']}")
    print(f"Flags    : {cfg['total_flags']}")
    print()

    env = MCPSecGymEnv(base_url=SERVER_URL).sync()
    results = []

    with env:
        for ep in range(num_episodes):
            stats = run_episode(env, task)
            results.append(stats)

            flag_str = f"{stats['flags_found']}/{cfg['total_flags']} flags"
            status = "SOLVED" if stats["completed"] else "      "
            print(
                f"  ep {ep + 1:02d} | "
                f"reward={stats['total_reward']:.3f} | "
                f"steps={stats['steps_taken']:02d} | "
                f"{flag_str} | {status}"
            )

    # Summary statistics
    rewards = [r["total_reward"] for r in results]
    steps = [r["steps_taken"] for r in results]
    flags = [r["flags_found"] for r in results]

    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)
    variance = sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)
    std_reward = variance**0.5
    avg_steps = sum(steps) / len(steps)
    flag_rate = sum(1 for f in flags if f >= 1) / len(flags) * 100
    solve_rate = sum(1 for r in results if r["completed"]) / len(results) * 100

    print()
    print(f"{'=' * 60}")
    print(f"SUMMARY [{task.upper()}]")
    print(f"{'=' * 60}")
    print(f"  avg_reward  : {avg_reward:.4f}")
    print(f"  std_reward  : {std_reward:.4f}")
    print(f"  max_reward  : {max_reward:.4f}")
    print(f"  min_reward  : {min_reward:.4f}")
    print(f"  avg_steps   : {avg_steps:.1f} / {cfg['max_steps']}")
    print(f"  flag_rate   : {flag_rate:.0f}%")
    print(f"  solve_rate  : {solve_rate:.0f}%")
    print()

    return {
        "task": task,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "flag_rate": flag_rate,
        "solve_rate": solve_rate,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Random agent evaluation for MCPSec Gym"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task to evaluate (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes per task (default: 20)",
    )
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    summaries = []

    for task in tasks:
        summary = run_eval(task, args.episodes)
        summaries.append(summary)

    if len(summaries) > 1:
        print(f"\n{'=' * 60}")
        print("CROSS-TASK COMPARISON")
        print(f"{'=' * 60}")
        print(f"  {'Task':<8} {'Avg Reward':>12} {'Std':>8} {'Flag%':>8} {'Solve%':>8}")
        print(f"  {'-' * 44}")
        for s in summaries:
            print(
                f"  {s['task']:<8} {s['avg_reward']:>12.4f} {s['std_reward']:>8.4f} "
                f"{s['flag_rate']:>7.0f}% {s['solve_rate']:>7.0f}%"
            )
        print()


if __name__ == "__main__":
    main()
