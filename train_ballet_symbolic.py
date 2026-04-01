import os
from pathlib import Path
import time
from argparse import ArgumentParser

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import flax.nnx as nnx
import gymnasium as gym
import gym_balletenv  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import orbax.checkpoint as ocp

from transformer_nnx import TransformerBackbone, TransformerConfig, TransformerState, reset_done_in_state
from ppo_core import (
    CategoricalCritic,
    ReplayBuffer,
    sample_action,
    bootstrap_value,
    calculate_gae,
    normalize_advantages,
    make_minibatches,
    update_ppo,
)


MODEL_DTYPE = jnp.bfloat16
PARAM_DTYPE = jnp.float32
NUM_LANG = 14
OBS_DIM = 121 + NUM_LANG

_LANG_EYE = np.eye(NUM_LANG, dtype=np.float32)


def preprocess_obs(obs_tuple):
    board, lang = obs_tuple  # (N, 121), (N,)
    lang_onehot = _LANG_EYE[lang]  # (N, 14)
    return np.concatenate([board, lang_onehot], axis=-1)  # (N, 135)


class PPOTransformerBalletSymbolic(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        transformer_cfg: TransformerConfig,
        critic: CategoricalCritic,
        *,
        rngs: nnx.Rngs,
    ):
        self.transformer_cfg = transformer_cfg
        self.critic = critic

        self.encoder = nnx.Linear(
            obs_dim,
            transformer_cfg.hidden_dim,
            dtype=MODEL_DTYPE,
            param_dtype=PARAM_DTYPE,
            rngs=rngs,
        )
        self.backbone = TransformerBackbone(transformer_cfg, rngs=rngs)
        self.policy_head = nnx.Linear(
            transformer_cfg.hidden_dim,
            num_actions,
            dtype=transformer_cfg.dtype,
            param_dtype=transformer_cfg.param_dtype,
            rngs=rngs,
        )
        self.critic_head = nnx.Linear(
            transformer_cfg.hidden_dim,
            critic.output_dim,
            dtype=transformer_cfg.dtype,
            param_dtype=transformer_cfg.param_dtype,
            rngs=rngs,
        )

    def _encode_obs(self, obs):
        return nnx.relu(self.encoder(jnp.asarray(obs, dtype=MODEL_DTYPE)))

    def init_state(self, batch_size: int) -> TransformerState:
        return self.backbone.init_state(batch_size, dtype=self.transformer_cfg.dtype)

    def step(self, obs, state: TransformerState):
        hidden = self._encode_obs(obs)
        next_state, hidden = self.backbone.step(hidden, state)
        logits = self.policy_head(hidden)
        critic = self.critic.predict(self.critic_head(hidden))
        return logits, critic, next_state

    def unroll(self, obs_seq, done_seq):
        hidden = self._encode_obs(obs_seq)
        hidden = self.backbone.unroll(jnp.swapaxes(hidden, 0, 1), jnp.swapaxes(done_seq, 0, 1))
        logits = jnp.swapaxes(self.policy_head(hidden), 0, 1)
        critic_logits = jnp.swapaxes(self.critic_head(hidden), 0, 1)
        return logits, self.critic.predict(critic_logits)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--level-name", type=str, default="6_delay48")
    parser.add_argument("--num-dancers-range", type=int, nargs="+", default=None)
    parser.add_argument("--dance-delay-range", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=100000)
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-minibatch", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=1e-2)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--multiple-of", type=int, default=256)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=500000.0)

    parser.add_argument("--critic-num-bins", type=int, default=1024)
    parser.add_argument("--critic-value-min", type=float, default=0.0)
    parser.add_argument("--critic-value-max", type=float, default=1.0)
    parser.add_argument("--critic-sigma", type=float, default=None)
    parser.add_argument("--save-ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--save-ckpt-every", type=int, default=0, help="save checkpoint every N global env steps (0=disabled)")
    return parser.parse_args()


def validate_args(args):
    if args.context_len < 1:
        raise ValueError(f"context_len must be >= 1, got {args.context_len}")
    if args.num_minibatch < 1:
        raise ValueError(f"num_minibatch must be >= 1, got {args.num_minibatch}")
    if args.num_envs % args.num_minibatch != 0:
        raise ValueError(f"num_envs must be divisible by num_minibatch, got {args.num_envs}, {args.num_minibatch}")
    if args.critic_num_bins < 2:
        raise ValueError(f"critic_num_bins must be >= 2, got {args.critic_num_bins}")
    if args.max_grad_norm <= 0.0:
        raise ValueError(f"max_grad_norm must be > 0, got {args.max_grad_norm}")
    if args.critic_value_min >= args.critic_value_max:
        raise ValueError(
            f"critic_value_min must be < critic_value_max, got {args.critic_value_min} and {args.critic_value_max}"
        )
    if args.critic_sigma is None:
        args.critic_sigma = 0.75 * ((args.critic_value_max - args.critic_value_min) / args.critic_num_bins)
    if args.critic_sigma <= 0.0:
        raise ValueError(f"critic_sigma must be > 0, got {args.critic_sigma}")


def main():
    args = parse_arguments()
    validate_args(args)
    envs_per_batch = args.num_envs // args.num_minibatch

    env_kwargs = {
        "level_name": args.level_name,
        "symbolic": True,
    }
    if args.num_dancers_range is not None:
        env_kwargs["num_dancers_range"] = args.num_dancers_range
    if args.dance_delay_range is not None:
        env_kwargs["dance_delay_range"] = args.dance_delay_range

    envs = gym.make_vec(
        "gym_balletenv/BalletEnvironment-v1",
        num_envs=args.num_envs,
        vectorization_mode="sync",
        **env_kwargs,
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    vector_env = envs.env if hasattr(envs, "env") else envs
    max_episode_steps = getattr(vector_env.envs[0].unwrapped, "max_episode_length", args.context_len)
    effective_context_len = max(args.context_len, max_episode_steps)

    rngs = nnx.Rngs(args.seed)
    model = PPOTransformerBalletSymbolic(
        obs_dim=OBS_DIM,
        num_actions=8,
        transformer_cfg=TransformerConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            context_len=effective_context_len,
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
            sigma=args.critic_sigma,
        ),
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adamw(args.learning_rate),
        ),
        wrt=nnx.Param,
    )
    metrics = nnx.metrics.MultiMetric(
        actor_loss=nnx.metrics.Average("actor_loss"),
        critic_loss=nnx.metrics.Average("critic_loss"),
        entropy=nnx.metrics.Average("entropy"),
    )

    wandb.init(
        project="ppo-llama-rl-nnx",
        name=f"ppo_transformer_ballet_symbolic_{args.level_name}",
        config={
            **vars(args),
            "requested_context_len": args.context_len,
            "effective_context_len": effective_context_len,
            "max_episode_steps": max_episode_steps,
        },
    )

    ckpt_mngr = None
    if args.save_ckpt_every > 0:
        ckpt_dir = (Path(args.save_ckpt_dir) / wandb.run.name).resolve()
        ckpt_mngr = ocp.CheckpointManager(ckpt_dir)

    replay_buffer = ReplayBuffer(effective_context_len, args.num_envs, (OBS_DIM,))

    global_env_step = 0
    start_time = time.time()
    for iteration in range(args.num_iter):
        obs = preprocess_obs(envs.reset(seed=args.seed + iteration * args.num_envs)[0])
        state = model.init_state(args.num_envs)
        done = np.zeros(args.num_envs, dtype=np.float32)
        rollout_rewards = []
        rollout_lengths = []

        for _ in range(effective_context_len):
            state_for_step = reset_done_in_state(state, done)
            log_prob, action, value, state = sample_action(model, obs, state_for_step, rngs)

            next_obs_tuple, reward, terminated, truncated, info = envs.step(np.asarray(action))
            next_done = np.maximum(terminated, truncated).astype(np.float32)

            replay_buffer.add(obs, action, log_prob, reward, done, value)
            global_env_step += args.num_envs

            if ckpt_mngr is not None and global_env_step % args.save_ckpt_every < args.num_envs:
                _, model_state = nnx.split(model)
                ckpt_mngr.save(global_env_step, args=ocp.args.Composite(
                    model=ocp.args.StandardSave(model_state),
                ))

            if "_episode" in info:
                for idx, finished in enumerate(info["_episode"]):
                    if finished:
                        rollout_rewards.append(float(info["episode"]["r"][idx]))
                        rollout_lengths.append(int(info["episode"]["l"][idx]))

            obs = preprocess_obs(next_obs_tuple)
            done = next_done

        obs_batch, actions_batch, log_probs_batch, rewards_batch, dones_batch, values_batch = replay_buffer.get()

        next_value = bootstrap_value(model, obs, reset_done_in_state(state, done))
        advantages, returns = calculate_gae(
            rewards_batch,
            values_batch,
            dones_batch,
            next_value,
            jnp.asarray(done, dtype=jnp.float32),
            gamma=args.gamma,
            lmbda=args.lmbda,
        )
        train_batch = (
            obs_batch,
            dones_batch,
            actions_batch,
            log_probs_batch,
            normalize_advantages(advantages),
            returns,
        )

        for _ in range(args.num_epochs):
            env_indices = np.asarray(jax.random.permutation(rngs(), args.num_envs))
            minibatches = make_minibatches(train_batch, env_indices, envs_per_batch)
            update_ppo(model, optimizer, minibatches, metrics, clip_eps=args.clip_eps, ent_coef=args.ent_coef)

        metric_values = {k: float(v) for k, v in metrics.compute().items()}
        sps = int(global_env_step / max(time.time() - start_time, 1e-6))
        log_data = {
            "train/iteration": iteration,
            "train/global_env_step": global_env_step,
            "train/sps": sps,
            "train/actor_loss": metric_values["actor_loss"],
            "train/critic_loss": metric_values["critic_loss"],
            "train/entropy": metric_values["entropy"],
        }
        if rollout_rewards:
            log_data["episode/reward_mean"] = float(np.mean(rollout_rewards))
            log_data["episode/reward_max"] = float(np.max(rollout_rewards))
            log_data["episode/length_mean"] = float(np.mean(rollout_lengths))
            log_data["episode/count"] = len(rollout_rewards)
        wandb.log(log_data, step=global_env_step)

        metrics.reset()
        replay_buffer.reset()

    if ckpt_mngr is not None:
        ckpt_mngr.wait_until_finished()

    envs.close()


if __name__ == "__main__":
    main()
