# ppo-llama-rl

PPO with LLaMA-style Transformer backbone (Flax NNX) for memory-demanding RL environments.

## Installation

```bash
pip install -r requirements.txt
wandb login
```


## Training

### BalletEnv (Symbolic)

```bash
# Default: 6 dancers, delay 48
python train_ballet_symbolic.py --level-name 6_delay48

# Custom difficulty with checkpoint saving
python train_ballet_symbolic.py \
    --level-name 5_delay48 \    # give a biggest number of dancers and delay
    --num-dancers-range 2 3 4 5 \
    --dance-delay-range 8 16 32 48 \
    --save-ckpt-dir ./checkpoints
```

### MemoryGym (Endless-MysteryPath)

```bash
# Default: Endless-MysteryPath-v0
python train_memorygym.py --env-name Endless-MysteryPath-v0

# With checkpoint saving
python train_memorygym.py \
    --env-name Endless-MysteryPath-v0 \
    --save-ckpt-dir checkpoints
```

## Evaluation

### BalletEnv (Symbolic)

```bash
python eval_ballet_symbolic.py \
    --checkpoint-dir checkpoints/<run-name> \
    --level-name 5_delay16 \
    --num-episodes 20 \
    --num-videos 5

# Evaluate a specific checkpoint step
python eval_ballet_symbolic.py \
    --checkpoint-dir checkpoints/<run-name> \
    --step 5000000 \
    --level-name 6_delay48
```

Results are saved to `eval_results/<run-name>/step_<step>/<level-name>/`:
- `results.json` : per-episode reward, length, and success rate summary
- `videos/` : rendered episode videos (mp4)

### MemoryGym (Endless-MysteryPath)

```bash
python eval_memorygym.py \
    --checkpoint-dir checkpoints/ <run-name> \
    --env-name Endless-MysteryPath-v0 \
    --max-steps 1024 \ # max steps in case of the environment is endless
    --num-episodes 20 \
    --num-videos 5

```

Results are saved to `eval_results/<run-name>/step_<step>/<env-name>/`:
- `results.json` : per-episode reward, length, and success rate summary
- `videos/` : rendered episode videos (mp4)
