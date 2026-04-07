"""
train.py — GRPO Training Script for MCPSec Gym

WHAT THIS FILE IS:
  This is where a language model actually *learns* to find security
  vulnerabilities. We use an algorithm called GRPO (Group Relative Policy
  Optimization) — a form of reinforcement learning designed specifically
  for training LLMs.

WHAT IS GRPO? (simple version)
  Imagine the model trying to solve an episode many times. Some attempts
  score high reward; some score low. GRPO asks: "compared to the average
  score of this batch, which attempts were above average?"

  It then adjusts the model's weights to:
    - Make high-reward actions more likely next time (positive gradient)
    - Make low-reward actions less likely next time (negative gradient)

  The key word is GROUP: rewards are compared within a group of attempts
  at the same prompt. This is what makes GRPO more stable than basic
  policy gradient — you're not chasing an absolute reward number, you're
  always comparing relative performance within the group.

HOW THE TRAINING LOOP WORKS:
  1. dataset      — a list of 3000 identical prompts ("Find vulnerabilities")
                    The prompt is just a seed; the real task is the environment.

  2. rollout_func — for each prompt, we run one full episode in MCPSec Gym.
                    The LLM generates tool calls; the env gives back rewards.
                    We collect all the token IDs and rewards.

  3. reward_funcs — functions that extract specific reward signals from the
                    episode data. We use three:
                      - reward_total:       total episode reward (main signal)
                      - reward_efficiency:  did the model find flags fast?
                      - reward_coverage:    did the model discover both vuln types?

  4. GRPOTrainer  — combines all of the above, runs the backward pass,
                    updates the model weights.

HOW THE LLM TAKES ACTIONS:
  Unlike a regular chatbot, the model here must output *structured JSON*
  on every turn. The format is:

    {"tool": "get_config", "parameters": {"section": "auth"}}

  We parse this JSON, convert it to an MCPSecAction, and send it to the
  environment. If the model outputs malformed JSON, we send a no-op action
  (which gets a small negative reward).

SYSTEM PROMPT DESIGN:
  The system prompt tells the model:
    - What the task is (find flags in a simulated app)
    - What tools are available
    - What JSON format to use for actions
  Good system prompt design is crucial — it shapes what actions the model
  even considers outputting.

HOW TO RUN:
  Requirements: a GPU with at least 16GB VRAM, plus:
    pip install trl[vllm] peft transformers datasets

  Option 1 — colocate vLLM with training (1 GPU):
    PYTHONPATH=. python3 train.py --vllm-mode colocate

  Option 2 — separate vLLM server (2 GPUs):
    # Terminal 1 (GPU 0)
    trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 9000

    # Terminal 2 (GPU 1)
    PYTHONPATH=. python3 train.py --vllm-mode server --vllm-server-url http://localhost:9000

  Make sure the MCPSec Gym server is also running:
    PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000 &
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

sys.path.insert(0, ".")

from client import MCPSecGymEnv
from models import MCPSecAction

# ---------------------------------------------------------------------------
# System Prompt
#
# This is what the model sees at the start of every episode.
# It defines the task, available tools, and output format.
# Think of it as the "rules of the game" given to the model.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert penetration tester performing authorized security testing.

Your goal: find all security vulnerabilities and capture all flags hidden in the system.

On every turn, respond with ONLY a single JSON object in this exact format:
  {"tool": "<tool_name>", "parameters": {<key>: <value>}}

Examples:
  {"tool": "get_config", "parameters": {"section": "auth"}}
  {"tool": "read_file", "parameters": {"path": "../../etc/passwd"}}
  {"tool": "search_files", "parameters": {"query": "secret"}}
  {"tool": "list_users", "parameters": {"role": "admin"}}
  {"tool": "query_db", "parameters": {"table": "users", "filter": "' OR '1'='1"}}

Do not include any explanation or text outside the JSON object.
Think carefully about what paths and config sections are most likely
to contain sensitive credentials, leaked secrets, or traversal vectors.
"""

# Task-specific configurations
TASK_CONFIG = {
    "easy": {"max_steps": 15, "total_flags": 2, "vuln_types": 2},
    "medium": {"max_steps": 20, "total_flags": 3, "vuln_types": 3},
    "hard": {"max_steps": 25, "total_flags": 3, "vuln_types": 5},
}

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO training for MCPSec Gym — teaches an LLM to find vulns."
    )

    # --- Environment ---
    parser.add_argument(
        "--env-url",
        default="http://localhost:8000",
        help="URL for the MCPSec Gym server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Which task to train on (default: easy).",
    )

    # --- Model ---
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model to train (default: Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Tokenizer ID. Defaults to --model-id if not set.",
    )

    # --- Dataset ---
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=3000,
        help=(
            "Number of training prompts. Each is an independent episode. "
            "More = more diverse training, but longer wall-clock time."
        ),
    )
    parser.add_argument(
        "--dataset-prompt",
        default="Find all security vulnerabilities and capture all flags.",
        help="The seed prompt for every training episode.",
    )

    # --- Training hyperparameters ---
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Max steps per episode. Defaults to task's MAX_STEPS (easy=15, medium=20, hard=25).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help=(
            "Max tokens the model generates per turn. "
            "64 is enough for a JSON tool call. "
            "Higher = slower but handles more complex reasoning."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help=(
            "How big each gradient step is. "
            "Too large → unstable training. Too small → slow convergence. "
            "5e-6 is a safe starting point for 1.5B models."
        ),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="How many times to loop over the dataset.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help=(
            "GRPO group size. For each prompt, the model attempts N episodes. "
            "Rewards are normalized within this group. "
            "Higher = more stable gradient, but N× more env calls."
        ),
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help=(
            "Simulate larger batch by accumulating gradients over N steps "
            "before doing a single optimizer update."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of steps to linearly increase the learning rate from 0.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help=(
            "Sampling temperature. Higher = more diverse completions "
            "(good for exploration). Lower = more deterministic."
        ),
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save checkpoints. Defaults to outputs/mcpsec-grpo-<timestamp>.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=20,
        help="Save a checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Log training metrics every N steps.",
    )

    # --- vLLM ---
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
        help=(
            "'colocate': run vLLM on the same GPU as training (1 GPU, slower). "
            "'server':   connect to a separate vLLM server (2 GPUs, faster)."
        ),
    )
    parser.add_argument(
        "--vllm-server-url",
        default="http://localhost:9000",
        help="URL for the vLLM inference server (only used with --vllm-mode=server).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Action Parsing
#
# The model outputs text. We need to turn that text into a valid MCPSecAction.
# This is always the trickiest part of LLM-as-agent training: the model
# doesn't always output perfect JSON.
# ---------------------------------------------------------------------------

# Fallback action used when the model outputs something we can't parse.
# Using "search_files" with a benign query gives a small informational reward
# rather than a hard error, which keeps gradients flowing.
NOOP_ACTION = MCPSecAction(tool_name="search_files", parameters={"query": "."})


def parse_action(text: str) -> MCPSecAction:
    """
    Parse the model's text output into an MCPSecAction.

    The model is *supposed* to output clean JSON like:
      {"tool": "get_config", "parameters": {"section": "auth"}}

    But in practice, it might:
      - Add explanation text before/after the JSON
      - Use single quotes instead of double quotes
      - Forget to include a tool name
      - Output partial JSON if max_new_tokens was too small

    We handle these gracefully rather than crashing.
    """
    text = text.strip()

    # Try to find a JSON block inside the text even if there's surrounding prose
    # e.g. "Let me check the config. {"tool": ...}"
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return NOOP_ACTION  # no JSON found at all

    json_str = text[start:end]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return NOOP_ACTION  # malformed JSON

    tool_name = data.get("tool") or data.get("tool_name")
    parameters = data.get("parameters") or data.get("params") or {}

    if not tool_name or not isinstance(tool_name, str):
        return NOOP_ACTION  # no tool name

    if not isinstance(parameters, dict):
        return NOOP_ACTION  # parameters must be a dict

    return MCPSecAction(tool_name=tool_name, parameters=parameters)


# ---------------------------------------------------------------------------
# Reward Functions
#
# GRPO takes a list of "reward functions". Each function receives the
# model's completions and any extra keyword arguments we pass through
# the rollout data, and returns a list of floats (one per completion).
#
# We use three reward signals:
#
#   1. total_reward     — the raw environment reward for the whole episode.
#                         This is the main signal. It rewards finding vulns
#                         and flags, and penalizes wasted steps.
#
#   2. efficiency_bonus — did the model find both flags in fewer steps?
#                         Encourages the model to be direct rather than
#                         exhaustively trying every parameter.
#
#   3. coverage_bonus   — did the model trigger both vuln types (info_leak
#                         and path_traversal)?  This rewards exploration:
#                         we don't want the model to learn one trick and
#                         stop.
#
# Each reward function's contribution is weighted by GRPOTrainer.
# ---------------------------------------------------------------------------


def reward_total(completions: list[str], **kwargs) -> list[float]:
    """
    Main reward: total environment reward across all steps of the episode.
    Range: roughly 0.0 (missed everything) to ~2.0 (captured both flags fast).
    """
    rewards = kwargs.get("total_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


def reward_efficiency(completions: list[str], **kwargs) -> list[float]:
    """
    Efficiency bonus: fraction of max steps that WEREN'T used.
    If the model solved both flags in 5 steps out of 15, this is (15-5)/15 = 0.67.
    If it ran out of steps without solving, this is 0.0.

    Why? We want the model to learn to reason about WHICH tool to call
    rather than trying every combination. An efficient solution requires
    understanding the system, not just exhaustive search.
    """
    rewards = kwargs.get("efficiency_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


def reward_coverage(completions: list[str], **kwargs) -> list[float]:
    """
    Coverage bonus: did the model find both distinct vuln types?
    +1.0 if both info_leak AND path_traversal were discovered.
    +0.5 if only one was found.
    0.0 if none.

    Why? Without this, the model might over-specialize on one vuln type
    (whichever is easiest to find) and ignore the other.
    """
    rewards = kwargs.get("coverage_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


# ---------------------------------------------------------------------------
# Rollout Function
#
# This is the heart of the training loop. For each prompt in the dataset,
# `rollout_func` runs one full episode in the environment and returns:
#   - prompt_ids:     tokenized input (the prompt given to the model)
#   - completion_ids: tokenized output (what the model generated)
#   - logprobs:       log-probabilities of each completion token
#   - reward fields:  the scalar reward signals for this episode
#
# The GRPOTrainer uses prompt_ids + completion_ids + logprobs to compute
# the policy gradient. The reward fields are passed to the reward functions.
# ---------------------------------------------------------------------------


def make_rollout_func(
    env: MCPSecGymEnv,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    max_turns: int,
    task: str = "easy",
):
    """
    Returns a rollout_func closure that captures the env and tokenizer.
    We return a closure (a function that remembers its outer variables)
    so we can pass it to GRPOTrainer cleanly.
    """

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        """
        Called by GRPOTrainer with a batch of prompts.
        For each prompt, we run one full episode and collect the trajectory.
        """
        # accumulators — one entry per episode in this batch
        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_total_rewards: list[float] = []
        all_efficiency_rewards: list[float] = []
        all_coverage_rewards: list[float] = []

        for prompt_text in prompts:
            # -----------------------------------------------------------------
            # Run one episode
            # -----------------------------------------------------------------
            result = env.reset(task=task)
            episode_total_reward = 0.0
            steps_taken = 0
            final_obs = result.observation

            # These accumulate across the whole episode
            ep_prompt_ids: list[int] = []
            ep_completion_ids: list[int] = []
            ep_logprobs: list[float] = []

            for _turn in range(max_turns):
                if result.done:
                    break

                # ---------------------------------------------------------
                # Build the prompt the model sees this turn.
                # It includes: system prompt, task description,
                # and the observation from the last step.
                # ---------------------------------------------------------
                obs = result.observation
                obs_text = (
                    f"Step {obs.step_count} of {obs.steps_remaining + obs.step_count}\n"
                    f"Available tools: {', '.join(obs.available_tools)}\n"
                    f"Flags captured: {obs.flags_captured}\n"
                    f"Vulns discovered: {obs.vulns_discovered}\n"
                    f"Alert level: {obs.alert_level}\n"
                    f"Last response: {obs.tool_response}\n\n"
                    f"Task: {prompt_text}\n"
                    "Your next action (JSON only):"
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_text},
                ]
                prompt_str = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )

                # ---------------------------------------------------------
                # Ask the model to generate a completion (tool call)
                # generate_rollout_completions handles the vLLM call
                # and returns token IDs + logprobs.
                # ---------------------------------------------------------
                rollout_output = generate_rollout_completions(trainer, [prompt_str])[0]
                ep_prompt_ids.extend(rollout_output["prompt_ids"])
                ep_completion_ids.extend(rollout_output["completion_ids"])
                ep_logprobs.extend(rollout_output["logprobs"])

                # Decode the model's output back to text
                completion_text = rollout_output.get("text") or tokenizer.decode(
                    rollout_output["completion_ids"], skip_special_tokens=True
                )

                # Parse the text into a structured action
                action = parse_action(completion_text)

                # Step the environment
                result = env.step(action)
                episode_total_reward += float(result.reward or 0.0)
                steps_taken += 1
                final_obs = result.observation

            # -----------------------------------------------------------------
            # Compute episode-level reward signals
            # -----------------------------------------------------------------
            task_cfg = TASK_CONFIG[task]

            # Efficiency: reward solving faster. Zero if never solved.
            solved = len(final_obs.flags_captured) == task_cfg["total_flags"]
            efficiency = (
                (task_cfg["max_steps"] - steps_taken) / task_cfg["max_steps"]
                if solved
                else 0.0
            )

            # Coverage: reward discovering all vuln types
            vuln_types = set(final_obs.vulns_discovered)
            coverage = len(vuln_types) / task_cfg["vuln_types"]

            all_prompt_ids.append(ep_prompt_ids)
            all_completion_ids.append(ep_completion_ids)
            all_logprobs.append(ep_logprobs)
            all_total_rewards.append(episode_total_reward)
            all_efficiency_rewards.append(efficiency)
            all_coverage_rewards.append(coverage)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "total_reward": all_total_rewards,
            "efficiency_reward": all_efficiency_rewards,
            "coverage_reward": all_coverage_rewards,
        }

    return rollout_func


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    tokenizer_id = args.tokenizer_id or args.model_id
    task = args.task
    task_cfg = TASK_CONFIG[task]
    max_turns = args.max_turns or task_cfg["max_steps"]

    print("=" * 60)
    print("MCPSec Gym — GRPO Training")
    print("=" * 60)
    print(f"  model      : {args.model_id}")
    print(f"  task       : {task}")
    print(f"  env url    : {args.env_url}")
    print(f"  max turns  : {max_turns}")
    print(f"  dataset    : {args.dataset_size} episodes")
    print(f"  epochs     : {args.num_epochs}")
    print(f"  generations: {args.num_generations} per prompt (GRPO group size)")
    print(f"  lr         : {args.learning_rate}")
    print(f"  vllm mode  : {args.vllm_mode}")
    print()

    # -------------------------------------------------------------------------
    # Tokenizer
    # The tokenizer turns text into token IDs (numbers). The model works in
    # token IDs; the tokenizer is what translates between human text and model
    # input/output.
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token  # needed for batched training

    # -------------------------------------------------------------------------
    # Environment client (sync wrapper around the async WebSocket client)
    # -------------------------------------------------------------------------
    env = MCPSecGymEnv(base_url=args.env_url).sync()

    # -------------------------------------------------------------------------
    # Dataset
    # GRPO needs a dataset of prompts to iterate over.
    # We use identical prompts because the "diversity" comes from the
    # environment resetting to a new random state each episode.
    # -------------------------------------------------------------------------
    dataset = Dataset.from_dict({"prompt": [args.dataset_prompt] * args.dataset_size})

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_slug = args.model_id.replace("/", "-")
    default_output_dir = Path("outputs") / f"mcpsec-grpo-{model_slug}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)

    # -------------------------------------------------------------------------
    # GRPOConfig — all training hyperparameters in one place
    #
    # Key parameters explained:
    #   use_vllm                — use vLLM for fast inference during rollouts
    #   vllm_mode               — "colocate" (1 GPU) or "server" (2 GPUs)
    #   num_generations         — GRPO group size (G in the paper)
    #   max_completion_length   — max new tokens per model turn
    #   gradient_accumulation   — effective batch = per_device × accum_steps
    # -------------------------------------------------------------------------
    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=(
            args.vllm_server_url if args.vllm_mode == "server" else None
        ),
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        warmup_steps=args.warmup_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_interval,
        temperature=args.temperature,
        report_to="none",  # change to "wandb" or "tensorboard" if you want logging
    )

    # -------------------------------------------------------------------------
    # Rollout function — see make_rollout_func above for detailed comments
    # -------------------------------------------------------------------------
    rollout_func = make_rollout_func(
        env=env,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
        task=task,
    )

    # -------------------------------------------------------------------------
    # GRPOTrainer — the main training object from TRL
    #
    # reward_funcs: list of functions. Each returns [float] per completion.
    #   GRPOTrainer sums all reward signals for each completion.
    #   You can weight them differently using reward_weights= if needed.
    # -------------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_total,  # main signal (range ~0.0 to ~2.0)
            reward_efficiency,  # speed bonus (range 0.0 to 1.0)
            reward_coverage,  # vuln diversity bonus (range 0.0 to 1.0)
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("Starting training...")
    print(f"Checkpoints will be saved to: {output_dir}")
    print()
    print("What to watch during training:")
    print("  train/reward        — should increase over time (good sign)")
    print("  train/reward_std    — high std = model is exploring (good early on)")
    print("  train/kl            — KL divergence from original model")
    print("                        if this explodes, reduce learning_rate")
    print()

    try:
        trainer.train()
    finally:
        env.close()

    print()
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print()
    print("Next: run python3 eval.py to compare trained model vs random baseline.")


if __name__ == "__main__":
    main()
