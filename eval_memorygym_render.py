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
import memory_gym  # noqa: F401
import numpy as np
from tqdm import tqdm

from transformer_nnx import TransformerConfig
from ppo_core import CategoricalCritic
from train_memorygym import PPOTransformerMemoryGym, MODEL_DTYPE, PARAM_DTYPE, SUPPORTED_ENVS
from eval_utils import load_checkpoint, greedy_action, save_video


def parse_arguments():
    parser = ArgumentParser(description="Evaluate trained PPO Transformer on MemoryGym env")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Orbax checkpoint directory")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--env-name", type=str, default="Endless-MysteryPath-v0")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode (default: env max or 1000)")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-len", type=int, default=512)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--multiple-of", type=int, default=256)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=500000.0)

    parser.add_argument("--critic-num-bins", type=int, default=101)
    parser.add_argument("--critic-value-min", type=float, default=0.0)
    parser.add_argument("--critic-value-max", type=float, default=13.0)
    parser.add_argument("--critic-sigma", type=float, default=None)
    return parser.parse_args()


def resolve_max_steps(args):
    max_episode_steps = gym.spec(args.env_name).max_episode_steps
    if args.max_steps is not None:
        return args.max_steps
    elif max_episode_steps is not None:
        return max_episode_steps
    else:
        return 1000


def build_model(args, obs_shape, num_actions, rngs):
    critic_sigma = args.critic_sigma
    if critic_sigma is None:
        critic_sigma = 0.75 * ((args.critic_value_max - args.critic_value_min) / args.critic_num_bins)
    return PPOTransformerMemoryGym(
        obs_shape=obs_shape,
        num_actions=num_actions,
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
    if args.env_name not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported env: {args.env_name}. Supported: {sorted(SUPPORTED_ENVS)}")

    max_steps = resolve_max_steps(args)

    tmp_env = gym.make(args.env_name, render_mode=None)
    obs_shape = tmp_env.observation_space.shape
    num_actions = tmp_env.action_space.n
    tmp_env.close()

    rngs = nnx.Rngs(args.seed)
    print(f"Building model (obs_shape={obs_shape}, num_actions={num_actions})...")
    model = build_model(args, obs_shape, num_actions, rngs)
    print(f"Loading checkpoint from {args.checkpoint_dir}")
    step = load_checkpoint(model, args.checkpoint_dir, args.step)
    print(f"Checkpoint loaded (step {step})")

    episodes = []
    all_frames = {}

    for ep in tqdm(range(args.num_episodes), desc="Evaluating"):
        env = gym.make(args.env_name, render_mode=args.render_mode)
        obs = env.reset(seed=args.seed + ep)[0]
        state = model.init_state(batch_size=1)

        frames = []
        total_reward = 0.0
        length = 0

        for _ in range(max_steps):
            action, state = greedy_action(model, obs[None], state)
            action_int = int(action.item())
            if ep < args.num_videos:
                frames.append(env.render())
            next_obs, reward, terminated, truncated, info = env.step(action_int)
            total_reward += float(reward)
            length += 1
            if terminated or truncated:
                break
            obs = next_obs

        env.close()
        episodes.append({"episode": ep, "reward": total_reward, "length": length})
        if frames:
            all_frames[ep] = frames

    ckpt_name = Path(args.checkpoint_dir).name
    output_dir = Path("eval_results") / ckpt_name / f"step_{step}" / args.env_name
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
        "env_name": args.env_name,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "max_steps": max_steps,
        "summary": {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)),
            "success_rate": float(np.mean([l >= max_steps for l in lengths])),
        },
        "episodes": episodes,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults ({args.num_episodes} episodes, max_steps={max_steps}):")
    print(f"  Reward: {results['summary']['reward_mean']:.4f} +/- {results['summary']['reward_std']:.4f}")
    print(f"  Length: {results['summary']['length_mean']:.1f} +/- {results['summary']['length_std']:.1f}")
    print(f"  Success rate: {results['summary']['success_rate']:.2%}")
    print(f"  Videos saved: {len(all_frames)}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
