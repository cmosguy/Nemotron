# Nemotron 3 Nano RL (GRPO) with your own data and your own agent harness

This kit goes with the video tutorial. It covers three things:

1. How the Nano 3 RL stage works: GRPO in NeMo RL, rewards from NeMo Gym environments, and the
   curriculum-ordered RL blend. Real samples from the blend are in `samples/`.
2. How to turn your agent trajectories into SFT data and into RL tasks, then blend those tasks
   into the Nano 3 blend (or run them as a second RL stage on top).
3. How to train through your proprietary harness, and what that requires.

Recipe source: `src/nemotron/recipes/nano3/stage2_rl/`. Docs: `docs/nemotron/nano3/rl.md`.

## Contents

| Path | What it is |
|---|---|
| `samples/nano3_rl_blend_samples.jsonl` | One real row per Nano 3 RL source (7 rows, 6 environments), DAPO/Skywork placeholders resolved |
| `samples/nano3_rl_blend_samples_placeholders_raw.jsonl` | The same DAPO/Skywork rows before resolution |
| `samples/example_harness_trajectories.jsonl` | Made-up harness trajectories in the canonical schema (one deliberately broken) |
| `scripts/peek_blend.py` | Samples the 6.9 GB blend over HTTP range requests and shows the curriculum |
| `scripts/convert_trajectories.py` | Trajectories → SFT rows (`sft`) and → Gym RL task rows (`rl`) |
| `scripts/blend_rl.py` | Interleaves your tasks into the Nano 3 blend without breaking its curriculum |
| `gym/my_task_env/` | Gym Resources Server skeleton: your reward function (+ optional tools) and config |
| `gym/my_harness_agent/` | Gym Agent Server adapter skeleton that runs your harness with training-token capture |
| `configs/grpo_nano3_custom_overlay.yaml` | NeMo RL config keys that add your environment to Nano 3 GRPO |

The scripts use only the Python standard library. The two `gym/` skeletons were tested against
`nemo-gym` 0.7.0 from PyPI: they import, have no unimplemented abstract methods, and the example
trajectories go through `transcript_to_response()` → `verify()` with rewards that match the harness's
own success labels (1, 0, 1, 1). The harness call itself is a stub you fill in. Gym's `main` branch
has moved some modules since 0.7.0 (for example `nemo_gym.openai_utils` → `nemo_gym.core`); match the
imports to the Gym version you deploy.

## 1. How the Nano 3 RL stage works

```
SFT checkpoint ──► NeMo RL (GRPO, Megatron backend) ◄──► vLLM (rollouts, colocated)
                          ▲                                   │ OpenAI-compatible HTTP
                          │ rewards                            ▼
                     NeMo Gym: agent server ──► model server ──► policy
                               │
                               └──► resources server: tools + verify() → reward
```

Each GRPO step (from `stage2_rl/config/default.yaml`):

- Takes 128 prompts (`grpo.num_prompts_per_step`) and generates 16 responses each
  (`num_generations_per_prompt`): 2,048 rollouts per step.
- Each rollout's row carries `agent_ref`, which routes it to one of six Gym environments. That
  environment runs the episode and returns a reward.
- Advantage = reward minus the mean of the *other 15* rewards for the same prompt
  (`use_leave_one_out_baseline: true`), normalized (`normalize_rewards: true`). No value network.
- Policy loss is PPO-style clipped with asymmetric bounds (`ratio_clip_min: 0.2`,
  `ratio_clip_max: 0.28`), token-level, no KL penalty (`reference_policy_kl_penalty: 0`),
  importance-sampling correction for vLLM vs Megatron probability mismatch.
- LR 3e-6 constant, MoE router frozen, 32 nodes × 8 GPUs at full scale (`tiny.yaml`: 1 node).

A consequence worth internalizing: a prompt where all 16 rollouts get the same reward (all pass or
all fail) has zero advantage and teaches nothing. That is why task difficulty and pass-rate
profiling matter so much.

## 2. The RL blend

`nvidia/Nemotron-3-Nano-RL-Training-Blend`: 93,244 rows, 6.92 GB, one `train.jsonl`.

| Source | Share | Environment (`agent_ref.name`) | Reward |
|---|---|---|---|
| Nemotron-RL-coding-competitive_coding | 0.25 | `code_gen_simple_agent` | unit tests pass (≤50 per problem) |
| Nemotron-RL-knowledge-mcqa | 0.20 | `mcqa_simple_agent` | boxed letter matches |
| Nemotron-RL-instruction_following | 0.17 | `instruction_following_simple_agent` | IFEval-style constraint checks |
| Skywork-OR1-RL-Data (math, no OmniMath) | 0.12 | `math_with_judge_simple_agent` | answer equivalence |
| DAPO-Math-17k | 0.10 | `math_with_judge_simple_agent` | answer equivalence |
| Nemotron-RL-agent-workplace_assistant | 0.10 | `workplace_assistant_simple_agent` | expected tool calls made |
| Nemotron-RL-instruction_following-structured_outputs | 0.05 | `structured_outputs_simple_agent` | JSON matches schema |

Every row has the same skeleton:

```json
{
  "responses_create_params": {"input": [{"role": "user", "content": "..."}], "tools": [...]},
  "agent_ref": {"type": "responses_api_agents", "name": "workplace_assistant_simple_agent"},
  "ground_truth": [{"name": "email_delete_email", "arguments": "{\"email_id\": \"00000259\"}"}],
  "pass_rate": 0.75, "pass_rate_total": 8, "pass_rate_passed": 6,
  "dataset": "nano_v3_sft_profiled_workbench"
}
```

- `responses_create_params` is the prompt only, in OpenAI Responses format. No answers: the policy
  generates those during training.
- `agent_ref` routes the row to an environment.
- Verifier fields (`ground_truth`, `expected_answer`, `verifier_metadata.unit_tests`, `schema_str`,
  `instruction_id_list` + `kwargs`) are what `verify()` checks against.
- `pass_rate` is how often NVIDIA's SFT checkpoint solved the task in 8 tries.

**Placeholders.** DAPO and Skywork rows ship as pointers (`_hf_placeholder: {row, question_template}`)
for licensing reasons. `nemotron nano3 data prep rl` (or the dataset's `create_nanov3_jsonl.py`)
fetches the original question and answer and fills them in. Compare the two files in `samples/`.

**Curriculum.** The file is ordered easy → hard and the recipe reads it in order
(`data.shuffle: False`). Sampling the file with `peek_blend.py`:

| Position in file | Mean pass rate |
|---|---|
| 0% | 0.65 |
| 25% | 0.38 |
| 37% | 0.36 |
| 72% | 0.20 |
| 87% | 0.26 |
| 89% | 0.23 |

The tech report describes it: tasks the SFT model already solves 100% of the time are dropped; domain
ratios stay fixed per batch; difficulty shifts from easy to hard over training.

## 3. Your data, step by step

### Step 1: map your logs to the canonical schema

Edit `adapt_record()` in `scripts/convert_trajectories.py` once. The canonical schema is OpenAI chat
format plus `task_id`, `outcome` and `verifier` (see the docstring and
`samples/example_harness_trajectories.jsonl`).

### Step 2: SFT data from successful trajectories

```bash
python scripts/convert_trajectories.py sft --input my_runs.jsonl --output my_sft.jsonl --min-score 1.0
```

Add the file to your SFT blend (`src/nemotron/recipes/nano3/stage1_sft/config/data_prep/`); local
paths are accepted next to `hf://` ones:

```json
{"name": "my-harness-sft", "path": "/lustre/you/my_sft.jsonl"}
```

Keep the stock SFT blend in the mix so the model keeps its general skills.

### Step 3: RL task rows from the same trajectories

```bash
python scripts/convert_trajectories.py rl --input my_runs.jsonl --output my_rl_tasks.jsonl \
    --agent-name my_harness_agent --dataset-name my_harness_v1
```

One row per task. The assistant turns are removed (the policy has to produce them), tool schemas are
converted to Responses format, and your `verifier` fields go to the top level of the row.

What makes a good RL task:
- It has a checkable outcome: a final state, an expected action, an exact answer, a test.
- It is neither always solved nor never solved by your SFT model.
- The reward checks the outcome, not the wording or the exact path taken.

### Step 4: write the verifier and profile pass rates

Put your reward logic in `gym/my_task_env/app.py` (`verify()` returns 0.0–1.0). Then, with your SFT
checkpoint served by vLLM, collect 8 rollouts per task:

```bash
gym env start --resources-server my_task_env --model-type vllm_model \
    --model-url http://localhost:8000/v1 --model my-sft-checkpoint --model-api-key EMPTY
gym eval run --no-serve --agent my_harness_agent --input my_rl_tasks.jsonl \
    --output profile_rollouts.jsonl --num-repeats 8
```

Write each task's mean reward into its row as `pass_rate` (`gym eval profile` reports per-task pass
rates). Check a handful of rollouts by hand: if a strong model scores near zero, the verifier is
probably wrong.

### Step 5: blend, or run on top

```bash
# 15% of each batch is yours, Nano3 curriculum kept
python scripts/blend_rl.py --nano3 train-full.jsonl --mine my_rl_tasks.jsonl --ratio 0.15 --out-dir blend/

# Second RL stage after a finished Nano3 RL run: mostly yours, plus a Nano3 replay slice
python scripts/blend_rl.py --nano3 train-full.jsonl --mine my_rl_tasks.jsonl --ratio 0.6 \
    --nano3-rows 20000 --max-repeats 3 --out-dir stage2b/
```

`blend_report.json` tells you how many of your prompts land in each GRPO step. Aim for at least 8.
Avoid `--ratio 1.0` unless you accept regressions elsewhere: the tech report found single-environment
RL can cause unrecoverable degradation on other benchmarks.

### Step 6: train

Start from your SFT checkpoint: set `policy.model_name` to its Hugging Face export, or set
`initial_checkpoint` to the Megatron checkpoint with `convert_initial_checkpoint_to_hf: true`
(see `config/tiny.yaml`). Add the keys from `configs/grpo_nano3_custom_overlay.yaml`.

```bash
uv run nemotron nano3 rl -c tiny --run YOUR-CLUSTER   # smoke test first
```

Watch: your environment's reward curve, `token_mult_prob_error` (should stay below ~2%), and, with a
wrapped harness, the masked-rollout fraction.

## 4. Can you use your proprietary harness?

Yes. NeMo Gym is built so the agent loop ("harness") is swappable. There are four ways to connect it:

| Path | You build | Trains the model? | Works with the pinned Nano 3 container? |
|---|---|---|---|
| A. Port the loop into a native Gym agent server | Agent server that calls Gym's model server `/v1/responses` | Yes, token ids come back inline | Yes |
| B. Wrap the harness as an Agent Server adapter (`gym/my_harness_agent/`) | Adapter + point the harness's OpenAI client at Gym | Yes, with training-token capture | No, needs Gym ≥ 0.6 |
| C. Remote Agent (your service exposes `/v1/responses`) | Nothing in Gym | No, scores only | Yes |
| D. Custom Environment Server | Episode protocol (e.g. simulated users) | Yes | Depends |

Path A is the most robust if your harness's logic (prompting, tool schemas, stop rules) can be
re-expressed in Python. Path B keeps your harness code unchanged.

**What training needs from the harness (paths A and B):**

1. Every model call goes to the Gym model server URL. The adapter gets it from
   `self.resolve_model_base_url(model_server_name, rollout_id)`; with capture on it looks like
   `http://<host>/ng-rollout/<id>/training-token-capture/v1`. Calls to anything else are not captured
   and the rollout is masked (`mask_sample: true`).
2. The conversation only grows. Each call's prompt must start with the previous prompt plus what the
   model sampled. Context compaction, summarization, history rewriting, or sub-agents create extra
   chains; Gym trains on one verified chain per rollout and masks ambiguous ones.
3. Tools go through the API (`tools=[...]`), parsed by vLLM's `qwen3_coder` parser in the Nano 3 config.
4. The full trajectory fits `policy.max_total_sequence_length` (49,152 tokens in the recipe).
5. Sampling is pinned with `sampling_overrides`, since harnesses often send their own temperature.

Check `_ng_token_capture` on the first rollouts: `n_calls` > 1 for tool-using tasks,
`terminal_attribution.chain` = `delivered`, `mask_sample` absent or false.

**Proprietary code.** Everything runs on your cluster. The adapter lives in your private fork of
NeMo Gym and imports your harness; nothing needs to be published.

**Version notes.** The recipe pins `nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano` (and the NeMo RL
`nano-v3` branch). Training on external harnesses with token capture arrived in NeMo Gym 0.6.0
(September 2026). For path B, use a NeMo RL release that bundles Gym ≥ 0.6 and carry the Nano 3 GRPO
settings over. Check the NeMo RL release notes for the matching container.

## References

- NeMo Gym: Choose a path for your agent — https://docs.nvidia.com/nemo/gym/main/agent-server
- Training on external agent harnesses — https://docs.nvidia.com/nemo/gym/main/tutorials/training-tutorials/external-agent-harnesses
- Add or update a harness — https://docs.nvidia.com/nemo/gym/main/agent-server/add-or-update-a-harness
- Single-step environment tutorial — https://docs.nvidia.com/nemo/gym/main/environment-tutorials/single-step-environment
- Nemotron 3 Nano GRPO on Gym — https://docs.nvidia.com/nemo/gym/main/model-recipes/nemotron-3-nano
- Tech report §3.2 — https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf
