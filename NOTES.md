# MCPSec Gym — Iteration Notes

Tracking every improvement round for the OpenEnv hackathon submission.

---

## Round 1: Initial Build (April 5-6)

**What we built:**
- Easy scenario (config_leak): 2 flags, 3 tools, path traversal + info leak
- Medium scenario (chain_reaction): 3 flags, 5 tools, IDOR → SQLi → info leak chain
- Hard scenario (fortress_breach): 3 flags, 7 tools, honeypot + rate limiter + lockout
- 12 episode variants total (4 per difficulty), intra-variant noise
- Deterministic test suite: 69 tests passing
- Random agent baseline (eval.py): avg_reward=0.13, flag_rate=15%
- Hybrid inference agent: easy=0.95-0.98, medium=0.96-0.98, hard=0.75-0.76
- HF Space deployed, openenv validate 6/6

**What was missing:**
- No LICENSE file
- README was technical but not a showcase
- Only ~4 git commits (looked thrown together)
- eval.py/train.py hardcoded for easy mode only
- server/__init__.py only exported easy environment

---

## Round 2: Competitive Analysis (April 7)

**How we did it:**
- Launched 3 parallel subagents to research winning environments, review our code, and study judging criteria
- Identified 20+ actionable improvements ranked by impact

**Key findings:**
- Winning repos have 50-109 commits, badges, architecture diagrams, demo GIFs
- Our recon bonus was firing on EVERY search_files call, not once per episode — total rewards could exceed 1.0 despite claiming [0.0, 1.0] range
- eval.py and train.py were hardcoded for easy mode only
- openenv.yaml was minimal (7 lines vs detailed task descriptions)
- requirements.txt used wrong package name vs pyproject.toml

---

## Round 3: Bug Fixes + Polish (April 7)

**Critical bugs fixed:**
1. Recon bonus guard — added `_recon_bonus_given` flag in all 3 environments so the bonus fires exactly once per episode
2. Reward clamping — added `max(min(reward, 1.0), -1.0)` in all `_make_obs` methods
3. Hard mode docstring — referenced "FLAG 3/4" but only 3 flags exist

**Infrastructure fixes:**
4. Added MIT LICENSE file
5. Fixed server/__init__.py to export all 3 environment classes
6. Fixed requirements.txt package name to match pyproject.toml
7. Removed stale BSD copyright header from pyproject.toml
8. Expanded openenv.yaml with task descriptions and evaluation baselines

**Feature improvements:**
9. eval.py now supports `--task easy|medium|hard|all` with cross-task comparison
10. train.py now supports `--task easy|medium|hard` with task-aware GRPO config

**Tests:** All 69 still passing after changes.

---

## Metrics Tracker

| Metric | Round 1 | Round 2 | Round 3 |
|--------|---------|---------|---------|
| Tests passing | 69/69 | 69/69 | 69/69 |
| openenv validate | 6/6 | 6/6 | TBD |
| Git commits | ~4 | ~4 | ~8 |
| HF Space status | UP | UP | TBD |
| Easy inference score | 0.95-0.98 | — | — |
| Medium inference score | 0.96-0.98 | — | — |
| Hard inference score | 0.75-0.76 | — | — |
| Known bugs | 2 critical | 2 critical | 0 |
