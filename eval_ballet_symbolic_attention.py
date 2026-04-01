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
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformer_nnx import TransformerConfig
from ppo_core import CategoricalCritic
from train_ballet_symbolic import PPOTransformerBalletSymbolic, MODEL_DTYPE, PARAM_DTYPE, NUM_LANG, OBS_DIM
from eval_utils import load_checkpoint

_LANG_EYE = np.eye(NUM_LANG, dtype=np.float32)

ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

LANG_NAMES = [
    "watch", "circle_cw", "circle_ccw", "up_and_down", "left_and_right",
    "diagonal_uldr", "diagonal_urdl", "plus_cw", "plus_ccw",
    "times_cw", "times_ccw", "zee", "chevron_down", "chevron_up",
]


def preprocess_obs(obs_tuple):
    """Preprocess single-env observation: (board, lang) -> flat array."""
    board, lang = obs_tuple
    board = np.asarray(board, dtype=np.float32)
    lang_onehot = _LANG_EYE[int(lang)]
    return np.concatenate([board, lang_onehot], axis=-1)


@nnx.jit
def greedy_action(model, obs, state):
    logits, _, next_state = model.step(obs, state)
    logits = logits.astype(jnp.float32)
    actions = jnp.argmax(logits, axis=-1)
    return actions, next_state


@nnx.jit
def greedy_action_with_attention(model, obs, state):
    logits, _, next_state, attn_list = model.step(obs, state, return_attention=True)
    logits = logits.astype(jnp.float32)
    actions = jnp.argmax(logits, axis=-1)
    return actions, next_state, attn_list


def parse_arguments():
    parser = ArgumentParser(description="Evaluate PPO Transformer on Ballet symbolic env with attention visualization")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Orbax checkpoint directory")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--level-name", type=str, default="5_delay16")
    parser.add_argument("--num-episodes", type=int, default=5)
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


def run_episode(model, env, seed, episode_dir):
    """Run one episode, saving observations and attention data.

    Returns dict with episode metadata.
    """
    obs_dir = episode_dir / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)

    obs_tuple, info = env.reset(seed=seed)
    raw_lang = int(obs_tuple[1])
    obs = preprocess_obs(obs_tuple)
    state = model.init_state(batch_size=1)

    total_reward = 0.0
    timestep = 0
    choice_phase_start = None
    choice_phase_lang_idx = None
    done = False

    while not done:
        # Save observation PNG
        frame = env.render()
        Image.fromarray(frame).save(obs_dir / f"step_{timestep:04d}.png")

        # Detect phase from raw language index
        is_choice_phase = raw_lang != 0

        if is_choice_phase:
            if choice_phase_start is None:
                choice_phase_start = timestep
                choice_phase_lang_idx = raw_lang
            action, state, attn_list = greedy_action_with_attention(model, obs[None], state)
            action_int = int(action.item())

            # Extract attention weights: list of [1, H, 1, S+1] -> [n_layers][H][valid_len]
            valid_len = int(state.valid_len[0])
            attn_data = []
            for layer_attn in attn_list:
                # layer_attn: [1, H, 1, S+1] -> [H, S+1]
                w = np.asarray(layer_attn[0, :, 0, :], dtype=np.float32)
                # Keep only the valid (most recent) entries
                # The cache is right-aligned: valid entries are the last valid_len positions
                attn_data.append(w[:, -valid_len:].tolist())

            attn_json = {
                "timestep": timestep,
                "action": action_int,
                "action_name": ACTION_NAMES[action_int],
                "reward": 0.0,  # will be updated after env.step
                "is_choice_phase": True,
                "attention_weights": attn_data,
            }
        else:
            action, state = greedy_action(model, obs[None], state)
            action_int = int(action.item())
            attn_json = None

        next_obs_tuple, reward, terminated, truncated, step_info = env.step(action_int)
        total_reward += float(reward)

        if attn_json is not None:
            attn_json["reward"] = float(reward)
            with open(episode_dir / f"attention_step_{timestep:04d}.json", "w") as f:
                json.dump(attn_json, f)

        done = terminated or truncated
        raw_lang = int(next_obs_tuple[1])
        obs = preprocess_obs(next_obs_tuple)
        timestep += 1

    metadata = {
        "total_steps": timestep,
        "choice_phase_start": choice_phase_start,
        "lang_idx": choice_phase_lang_idx,
        "lang_name": LANG_NAMES[choice_phase_lang_idx] if choice_phase_lang_idx is not None else None,
        "total_reward": total_reward,
        "level_name": env.spec.kwargs.get("level_name", "unknown") if env.spec else "unknown",
    }
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    args = parse_arguments()
    rngs = nnx.Rngs(args.seed)
    print("Building model...")
    model = build_model(args, rngs)
    print(f"Loading checkpoint from {args.checkpoint_dir}")
    step = load_checkpoint(model, args.checkpoint_dir, args.step)
    print(f"Checkpoint loaded (step {step})")

    env_kwargs = {"level_name": args.level_name, "symbolic": True, "render_mode": "rgb_array"}

    ckpt_name = Path(args.checkpoint_dir).name
    output_dir = Path("eval_results") / ckpt_name / f"step_{step}" / args.level_name
    attention_dir = output_dir / "attention"

    episode_names = []
    all_metadata = []

    for ep in tqdm(range(args.num_episodes), desc="Evaluating"):
        env = gym.make("gym_balletenv/BalletEnvironment-v1", **env_kwargs)
        ep_name = f"episode_{ep:03d}"
        episode_dir = attention_dir / ep_name

        metadata = run_episode(model, env, seed=args.seed + ep, episode_dir=episode_dir)
        episode_names.append(ep_name)
        all_metadata.append(metadata)
        env.close()

        print(f"  Episode {ep}: reward={metadata['total_reward']:.1f}, "
              f"steps={metadata['total_steps']}, "
              f"choice_start={metadata['choice_phase_start']}, "
              f"target={metadata['lang_name']}")

    # Write index.json
    with open(attention_dir / "index.json", "w") as f:
        json.dump({"episodes": episode_names}, f, indent=2)

    # Write summary results.json
    rewards = [m["total_reward"] for m in all_metadata]
    results = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "step": int(step),
        "level_name": args.level_name,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "summary": {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "success_rate": float(np.mean([r > 0 for r in rewards])),
        },
        "episodes": all_metadata,
    }
    results_path = output_dir / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Copy HTML visualization
    html_src = Path(__file__).parent / "visualize_attention.html"
    if html_src.exists():
        import shutil
        shutil.copy2(html_src, output_dir / "visualize_attention.html")
        print(f"\nVisualization: {output_dir / 'visualize_attention.html'}")
        print(f"  Serve with: cd {output_dir} && python -m http.server 8080")

    print(f"\nResults ({args.num_episodes} episodes):")
    print(f"  Reward: {results['summary']['reward_mean']:.4f} +/- {results['summary']['reward_std']:.4f}")
    print(f"  Success rate: {results['summary']['success_rate']:.2%}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
