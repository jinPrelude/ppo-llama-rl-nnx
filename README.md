# ppo-llama-rl

PPO with LLaMA-style Transformer backbone (Flax NNX) for memory-demanding RL environments.

## Project Structure

```
core/                   # Shared modules
  transformer_nnx.py    # LLaMA-style Transformer backbone
  ppo_core.py           # PPO algorithm
  eval_utils.py         # Checkpoint loading, greedy action, video saving
envs/                   # Environment-specific code
  memorygym/            # MemoryGym environments
    train_transformer.py
    eval_batch.py
    eval_render.py
    plot_mysterypath.py
```

See [docs/CONVENTIONS.md](docs/CONVENTIONS.md) for naming conventions and how to add new environments/backbones.

## Installation

```bash
pip install -r requirements.txt
wandb login
```

## Training (MemoryGym)

Trained on RTX6000 PRO + 32 CPUs. Total training time: 1 day 9 hours.

```bash
python -m envs.memorygym.train_transformer --env-name Endless-MysteryPath-v0 --context-len 512 --save-ckpt-every 5_000_000 --learning-rate 0.0001
```

### Training Curves

<table>
  <tr>
    <td><img src="imgs/reward_mean_steps.png" width="400"></td>
    <td><img src="imgs/length_mean_steps.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="imgs/reward_mean_wallclock.png" width="400"></td>
    <td><img src="imgs/sps.png" width="400"></td>
  </tr>
</table>

## Evaluation (MemoryGym)

| Script | Purpose | Parallelism | Output |
|--------|---------|-------------|--------|
| `envs/memorygym/eval_batch.py` | Fast stats collection | `AsyncVectorEnv` (parallel envs) | `batch_results.json` + survival rates |
| `envs/memorygym/eval_render.py` | Video rendering | Sequential (single env) | `results.json` + mp4 videos |

### Batch evaluation (fast, no video)

```bash
python -m envs.memorygym.eval_batch \
    --checkpoint-dir checkpoints/<run-name> \
    --env-name Endless-MysteryPath-v0 \
    --num-episodes 200 \
    --batch-size 50 \
    --max-steps 2048
```

### Render evaluation (with video)

```bash
python -m envs.memorygym.eval_render \
    --checkpoint-dir checkpoints/<run-name> \
    --env-name Endless-MysteryPath-v0 \
    --num-episodes 20 \
    --num-videos 5 \
    --max-steps 1024
```

Results are saved to `eval_results/<run-name>/step_<step>/<env-name>/`.

### Plotting

```bash
python -m envs.memorygym.plot_mysterypath --results-path eval_results/<run-name>/step_<step>/<env-name>/batch_results.json
```

Generates a survival rate vs step threshold curve from batch eval results.

![Survival Rate vs Step Threshold](imgs/mysterypath_survival.png)
