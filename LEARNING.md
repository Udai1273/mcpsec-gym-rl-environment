# Learning Notes — Building RL Environments

Everything we learned building MCPSec Gym that applies to building future
OpenEnv / RL environments. Written for our future selves.

---

## 1. Reward Design

**The hardest part of the whole project.** Get this wrong and nothing else matters.

### Principles that worked:
- **Non-sparse rewards**: Give signal at every meaningful step, not just at the end. A random agent getting 0.0 on every episode means training will never converge.
- **Reward budget = 1.0**: Plan your reward components so max possible = 1.0. Write out the math explicitly in comments. Easy to audit, easy to verify.
- **Decompose into components**: vulnerability discovery (+X), flag capture (+Y), efficiency bonus (+Z), penalties (-W). Each component rewards a different skill.
- **Clamp server-side**: Even if you're confident the math adds up, clamp in `_make_obs`. Bugs happen. Clamping is cheap insurance.

### Mistakes we made:
- Recon bonus wasn't guarded to fire once. A search-spamming agent could accumulate infinite 0.05 bonuses. Always use a boolean guard for "once per episode" rewards.
- Didn't clamp server-side initially. README said [0.0, 1.0] but the code didn't enforce it.

### Design pattern:
```python
reward = 0.0
if new_vuln:        reward += VULN_REWARD
if new_flag:        reward += FLAG_REWARD
if recon and not self._recon_given:
    reward += RECON_REWARD
    self._recon_given = True
if done and solved:  reward += efficiency_bonus
if done and alert:   reward -= ALERT_PENALTY
# Always clamp
return max(min(reward, 1.0), -1.0)
```

---

## 2. Episode Variant Design

**How to get dynamic problem generation right.**

### The pattern: Episode Variant Pool
- Define N variant dicts (we used 4 per difficulty = 12 total)
- Each variant specifies: which section leaks, which file to traverse, which table has the SQLi, etc.
- On reset(), pick one variant with `self._rng.choice(variants)`
- Generate fresh flags per episode: `f"FLAG{{{uuid.UUID(int=self._rng.getrandbits(128)).hex[:8]}}}"`

### Intra-variant noise
Even within a single variant, randomize:
- Traversal depth (2-4 levels of `../`)
- Number and selection of decoy config sections
- IDOR user IDs within a range
- JWT algorithm (HS256 vs HS512)

This means even if the agent memorizes "variant A always needs section=auth", the exact traversal path or user ID differs.

### RNG discipline
- **NEVER use global `random` module.** Always `self._rng = random.Random(seed)`.
- This makes episodes reproducible for testing: same seed = same variant, same flags.
- Tests use specific seeds and verify exact behavior.

---

## 3. Testing Strategy

### Seed-agnostic testing via `info["variant_id"]`
- `reset(seed=X)` returns `(obs, info)` where `info["variant_id"]` tells you which variant was chosen
- Tests call reset, read the variant_id, then know exactly what actions should work
- This means tests are deterministic but don't hardcode "seed 42 = variant A"

### Test structure (for each difficulty):
1. **Variant coverage**: verify each seed maps to the expected variant
2. **Flag capture**: optimal trajectory finds all flags, reward > threshold
3. **Negative tests**: wrong tool name gets -0.05, wrong traversal depth gets "not found"
4. **Boundary tests**: episode ends after MAX_STEPS, step_count increments correctly
5. **Defense tests** (medium/hard): rate limiter triggers at threshold, honeypot raises alert
6. **Reward budget**: verify max possible reward doesn't exceed 1.0

### The `> 0.5` threshold
- Tests check `total_reward > 0.5` not `== 1.0`
- Exact reward depends on floating point of efficiency bonus
- Threshold catches regressions without being brittle

---

## 4. Inference Agent Design

### The hybrid approach
Pure LLM inference is slow and unreliable. Our solution:
1. Use deterministic strategies for known patterns (easy/medium have predictable optimal paths)
2. Fall back to LLM only when deterministic strategy fails or for hard mode reasoning
3. This gives us 0.95+ on easy/medium and ~0.75 on hard

### Why pure LLM struggled:
- LLMs repeat the same tool call (hit rate limiter)
- LLMs don't naturally try path traversal depths
- LLMs don't chain findings (IDOR → SQLi → config leak requires 3-step reasoning)

---

## 5. Server Architecture

### FastAPI multi-task dispatcher
- Single server exposes `/reset`, `/step`, `/state`, `/health`
- `task` parameter in reset routes to the correct environment class
- Each environment class is independent, no shared state

### Concurrent sessions
- `SUPPORTS_CONCURRENT_SESSIONS = True`
- ALL state lives in `self._` attributes, never module globals
- This is critical for HF Spaces where multiple users hit the same server

---

## 6. Hackathon Presentation

### What judges actually look at (based on competitive analysis):
1. **Does openenv validate pass?** (table stakes — 6/6 or nothing)
2. **Is the HF Space alive?** (POST /reset returns 200)
3. **Is the README a showcase?** Badges, architecture diagram, "Why This Matters" section
4. **Commit history depth** — 50+ commits signals serious work, 3 commits signals weekend hack
5. **Does the environment actually train?** Showing before/after GRPO curves is a differentiator

### What we learned too late:
- Small, frequent commits from the start would have been free quality signal
- A demo GIF in the README is worth 1000 words of documentation
- LICENSE file should be the first file in any repo

---

## 7. OpenEnv Framework Specifics

### Required contract:
- `Environment` class with `reset(seed, **kwargs)` and `step(action, **kwargs)`
- `reset` returns `(observation, info_dict)`
- `step` returns an observation object
- `State` object with `episode_id` and `step_count`
- `openenv.yaml` with `spec_version`, `name`, `type`, `runtime`, `app`, `port`

### Docker deployment (HF Spaces):
- `server/Dockerfile` must install from `server/requirements.txt`
- App must listen on port 8000
- Health check: `GET /health` returns 200

### openenv validate checks:
1. YAML schema valid
2. Server starts
3. Health endpoint responds
4. Reset returns valid observation
5. Step returns valid observation
6. State endpoint works

---

## 8. Common Pitfalls

| Pitfall | How we hit it | Fix |
|---------|--------------|-----|
| Global RNG | Used `random.choice()` in early prototypes | `self._rng = random.Random(seed)` |
| Reward exceeds budget | Recon bonus fired every step | Boolean guard + server-side clamp |
| Tests pass but env is broken | Tests only covered easy mode | Test all 3 difficulties |
| Train script hardcoded | Only worked for easy task | Add `--task` flag with task config dict |
| Package name mismatch | requirements.txt vs pyproject.toml | Audit both files, keep in sync |
| Missing LICENSE | Copyright header referenced nonexistent file | Add LICENSE in first commit |
