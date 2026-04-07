"""
test_medium.py — Seed-agnostic tests for the medium scenario (chain_reaction).
"""

import pytest
from server.app import MCPSecMultiTaskEnvironment
from models import MCPSecAction


@pytest.fixture
def env():
    e = MCPSecMultiTaskEnvironment()
    yield e


# Golden paths: 3-step exploit chain per variant.
# Step 1: list_users(role=<idor_role>) → IDOR, flag_1
# Step 2: query_db with SQLi on the variant's sqli_table → flag_2
# Step 3: get_config on the variant's leaky section → flag_3
GOLDEN_PATHS_MEDIUM = {
    "chain_reaction_A": [
        {"tool_name": "list_users", "parameters": {"role": "admin"}},
        {
            "tool_name": "query_db",
            "parameters": {"table": "users", "filter": "' OR '1'='1"},
        },
        {"tool_name": "get_config", "parameters": {"section": "admin_keys"}},
    ],
    "chain_reaction_B": [
        {"tool_name": "list_users", "parameters": {"role": "devops"}},
        {
            "tool_name": "query_db",
            "parameters": {"table": "credentials", "filter": "' OR '1'='1"},
        },
        {"tool_name": "get_config", "parameters": {"section": "internal_tokens"}},
    ],
    "chain_reaction_C": [
        {"tool_name": "list_users", "parameters": {"role": "sysadmin"}},
        {
            "tool_name": "query_db",
            "parameters": {"table": "sessions", "filter": "' OR '1'='1"},
        },
        {"tool_name": "get_config", "parameters": {"section": "service_accounts"}},
    ],
    "chain_reaction_D": [
        {"tool_name": "list_users", "parameters": {"role": "superuser"}},
        {
            "tool_name": "query_db",
            "parameters": {"table": "audit_log", "filter": "' OR '1'='1"},
        },
        {"tool_name": "get_config", "parameters": {"section": "api_secrets"}},
    ],
}


@pytest.mark.parametrize("seed", range(20))
def test_medium_any_variant(env, seed):
    obs, info = env.reset(seed=seed, task="medium")
    variant_id = info["variant_id"]
    assert variant_id in GOLDEN_PATHS_MEDIUM, (
        f"Unknown variant_id '{variant_id}' — update GOLDEN_PATHS_MEDIUM"
    )

    actions = GOLDEN_PATHS_MEDIUM[variant_id]
    total_reward = 0.0
    for action_dict in actions:
        action = MCPSecAction(**action_dict)
        result = env.step(action)
        if isinstance(result, tuple):
            step_obs, reward, *_ = result
        else:
            step_obs = result
            reward = step_obs.reward
        total_reward += reward
        if step_obs.done:
            break

    assert total_reward > 0.5, (
        f"Variant {variant_id} (seed={seed}): reward={total_reward:.4f}, expected > 0.5"
    )


def test_medium_rate_limiter_raises_alert(env):
    """Calling the same tool 4 times in a row must raise alert level."""
    env.reset(seed=0, task="medium")
    for _ in range(4):
        action = MCPSecAction(tool_name="search_files", parameters={"query": "app"})
        result = env.step(action)
        obs = result[0] if isinstance(result, tuple) else result
    assert obs.alert_level >= 1


def test_medium_different_seeds_produce_different_variants(env):
    seen = set()
    for seed in range(20):
        _, info = env.reset(seed=seed, task="medium")
        seen.add(info["variant_id"])
    assert len(seen) >= 2


def test_medium_same_seed_reproducible(env):
    _, info_a = env.reset(seed=7, task="medium")
    _, info_b = env.reset(seed=7, task="medium")
    assert info_a["variant_id"] == info_b["variant_id"]
