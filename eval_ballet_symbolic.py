import json
import os
from argparse import ArgumentParser
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import flax.nnx as nnx
import gymnasium as gym
import gym_balletenv  # noqa: F401
import numpy as np
from tqdm import tqdm

from transformer_nnx import TransformerConfig
from ppo_core import CategoricalCritic
from train_ballet_symbolic import PPOTransformerBalletSymbolic, MODEL_DTYPE, PARAM_DTYPE, NUM_LANG, OBS_DIM
from eval_utils import load_checkpoint, greedy_action, save_video

_LANG_EYE = np.eye(NUM_LANG, dtype=np.float32)


def preprocess_obs(obs_tuple):
    """Preprocess single-env observation: (board, lang) -> flat array."""
    board, lang = obs_tuple
    board = np.asarray(board, dtype=np.float32)
    lang_onehot = _LANG_EYE[int(lang)]
    return np.concatenate([board, lang_onehot], axis=-1)


def parse_arguments():
    parser = ArgumentParser(description="Evaluate trained PPO Transformer on Ballet symbolic env")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Orbax checkpoint directory")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--level-name", type=str, default="5_delay16")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-len", type=int, default=2048)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--multiple-of", type=int, default=256)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=500000.0)

    parser.add_argument("--critic-num-bins", type=int, default=51)
    parser.add_argument("--critic-value-min", type=float, default=0.0)
    parser.add_argument("--critic-value-max", type=float, default=1.0)
    parser.add_argument("--critic-sigma", type=float, default=None)
    return parser.parse_args()


def build_model(args, rngs):
    critic_sigma = args.critic_sigma
    if critic_sigma is None:
        critic_sigma = 0.75 * ((args.critic_value_max - args.critic_value_min) / args.critic_num_bins)
    return PPOTransformerBalletSymbolic(
        obs_dim=OBS_DIM,
        num_actions=8,
        transformer_cfg=TransformerConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            context_len=args.context_len,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            norm_eps=args.norm_eps,
            rope_theta=args.rope_theta,
            dtype=MODEL_DTYPE,
            param_dtype=PARAM_DTYPE,
        ),
        critic=CategoricalCritic(
            num_bins=args.critic_num_bins,
            value_min=args.critic_value_min,
            value_max=args.critic_value_max,
            sigma=critic_sigma,
        ),
        rngs=rngs,
    )


def main():
    args = parse_arguments()
    rngs = nnx.Rngs(args.seed)
    print(f"Building model...")
    model = build_model(args, rngs)
    print(f"Loading checkpoint from {args.checkpoint_dir}")
    step = load_checkpoint(model, args.checkpoint_dir, args.step)
    print(f"Checkpoint loaded (step {step})")

    env_kwargs = {"level_name": args.level_name, "symbolic": True, "render_mode": "rgb_array"}

    episodes = []
    all_frames = {}

    for ep in tqdm(range(args.num_episodes), desc="Evaluating"):
        env = gym.make("gym_balletenv/BalletEnvironment-v1", **env_kwargs)
        obs = preprocess_obs(env.reset(seed=args.seed + ep)[0])
        state = model.init_state(batch_size=1)

        frames = []
        total_reward = 0.0
        length = 0

        done = False
        while not done:
            action, state = greedy_action(model, obs[None], state)
            action_int = int(action.item())
            if ep < args.num_videos:
                frames.append(env.render())
            next_obs_tuple, reward, terminated, truncated, info = env.step(action_int)
            total_reward += float(reward)
            length += 1
            done = terminated or truncated
            obs = preprocess_obs(next_obs_tuple)

        env.close()
        episodes.append({"episode": ep, "reward": total_reward, "length": length})
        if frames:
            all_frames[ep] = frames

    ckpt_name = Path(args.checkpoint_dir).name
    output_dir = Path("eval_results") / ckpt_name / f"step_{step}" / args.level_name
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for ep, frames in all_frames.items():
        video_path = videos_dir / f"episode_{ep:03d}.mp4"
        save_video(frames, str(video_path), args.fps)
        episodes[ep]["video"] = f"videos/episode_{ep:03d}.mp4"

    for ep_data in episodes:
        if "video" not in ep_data:
            ep_data["video"] = None

    rewards = [e["reward"] for e in episodes]
    lengths = [e["length"] for e in episodes]
    results = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "step": int(step),
        "level_name": args.level_name,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "summary": {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)),
            "success_rate": float(np.mean([r > 0 for r in rewards])),
        },
        "episodes": episodes,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults ({args.num_episodes} episodes):")
    print(f"  Reward: {results['summary']['reward_mean']:.4f} +/- {results['summary']['reward_std']:.4f}")
    print(f"  Length: {results['summary']['length_mean']:.1f} +/- {results['summary']['length_std']:.1f}")
    print(f"  Success rate: {results['summary']['success_rate']:.2%}")
    print(f"  Videos saved: {len(all_frames)}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
