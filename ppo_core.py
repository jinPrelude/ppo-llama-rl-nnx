import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import optax
import rlax
from flax import struct


@struct.dataclass
class CriticPrediction:
    logits: jax.Array
    value: jax.Array


@struct.dataclass
class CategoricalCritic:
    num_bins: int
    value_min: float
    value_max: float
    sigma: float

    @property
    def output_dim(self):
        return self.num_bins

    @property
    def bin_width(self):
        return (self.value_max - self.value_min) / self.num_bins

    @property
    def bin_edges(self):
        return jnp.linspace(self.value_min, self.value_max, self.num_bins + 1, dtype=jnp.float32)

    @property
    def bin_centers(self):
        edges = self.bin_edges
        return 0.5 * (edges[:-1] + edges[1:])

    def decode(self, logits):
        probs = jax.nn.softmax(logits, axis=-1)
        centers = self.bin_centers.astype(logits.dtype)
        return jnp.sum(probs * centers, axis=-1)

    def predict(self, logits):
        return CriticPrediction(logits=logits, value=self.decode(logits))

    def target_probs(self, target):
        target = jnp.clip(target, self.value_min, self.value_max)
        support = self.bin_edges.astype(target.dtype)
        cdf = jax.scipy.special.erf((support - target[..., None]) / (jnp.sqrt(2.0) * self.sigma))
        mass = cdf[..., 1:] - cdf[..., :-1]
        return mass / (cdf[..., -1] - cdf[..., 0])[..., None]

    def loss(self, prediction, target_value):
        target_probs = jax.lax.stop_gradient(self.target_probs(target_value))
        log_probs = jax.nn.log_softmax(prediction.logits, axis=-1)
        return -(target_probs * log_probs).sum(axis=-1).mean()


class ReplayBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_shape, obs_dtype=np.float32):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dtype = obs_dtype
        self.obs = np.zeros((num_steps, num_envs, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.reset()

    def reset(self):
        self.size = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        if self.size >= self.num_steps:
            raise ValueError("ReplayBuffer is full. Call reset() before adding new data.")
        t = self.size
        self.obs[t] = np.asarray(obs, dtype=self.obs_dtype)
        self.actions[t] = np.asarray(actions, dtype=np.int32)
        self.log_probs[t] = np.asarray(log_probs, dtype=np.float32)
        self.rewards[t] = np.asarray(rewards, dtype=np.float32)
        self.dones[t] = np.asarray(dones, dtype=np.float32)
        self.values[t] = np.asarray(values, dtype=np.float32)
        self.size += 1

    def get(self):
        if self.size != self.num_steps:
            raise ValueError(f"ReplayBuffer not full: expected {self.num_steps}, got {self.size}")
        return (
            jnp.asarray(self.obs),
            jnp.asarray(self.actions),
            jnp.asarray(self.log_probs),
            jnp.asarray(self.rewards),
            jnp.asarray(self.dones),
            jnp.asarray(self.values),
        )


@nnx.jit
def sample_action(model, obs, state, rngs):
    logits, critic, next_state = model.step(obs, state)
    logits = logits.astype(jnp.float32)
    value = critic.value.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions = rngs.categorical(logits, axis=-1)
    sampled_log_prob = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)
    return sampled_log_prob, actions, value, next_state


@nnx.jit
def bootstrap_value(model, obs, state):
    _, critic, _ = model.step(obs, state)
    return critic.value.astype(jnp.float32)


def calculate_gae(rewards, values, dones, next_value, next_done, gamma: float, lmbda: float):
    next_values = jnp.concatenate([values[1:], next_value[None, :]], axis=0)
    next_nonterminal = 1.0 - jnp.concatenate([dones[1:], next_done[None, :]], axis=0)
    deltas = rewards + gamma * next_values * next_nonterminal - values

    def scan_step(last_advantage, inputs):
        delta_t, nonterminal_t = inputs
        advantage = delta_t + gamma * lmbda * nonterminal_t * last_advantage
        return advantage, advantage

    init_advantage = jnp.zeros_like(next_value)
    _, advantages = jax.lax.scan(scan_step, init_advantage, (deltas, next_nonterminal), reverse=True)
    return advantages, advantages + values


def normalize_advantages(advantages):
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def loss_fn(model, batch, clip_eps, ent_coef):
    obs, dones, actions, old_log_probs, advantages, returns = batch
    logits, critic = model.unroll(obs, dones)

    logits = logits.astype(jnp.float32)
    critic = CriticPrediction(
        logits=critic.logits.astype(jnp.float32),
        value=critic.value.astype(jnp.float32),
    )
    old_log_probs = old_log_probs.astype(jnp.float32)
    advantages = advantages.astype(jnp.float32)
    returns = returns.astype(jnp.float32)
    clip_eps = jnp.asarray(clip_eps, dtype=jnp.float32)
    ent_coef = jnp.asarray(ent_coef, dtype=jnp.float32)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected_log_probs = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)
    ratio = jnp.exp(selected_log_probs - old_log_probs)

    actor_loss = rlax.clipped_surrogate_pg_loss(ratio.reshape(-1), advantages.reshape(-1), clip_eps).mean()
    critic_loss = model.critic.loss(critic, returns)
    entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * log_probs, axis=-1).mean()
    total_loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
    return total_loss, (actor_loss, critic_loss, entropy)


@nnx.jit
def update_ppo(model: nnx.Module, optimizer: nnx.Optimizer, minibatches, metrics: nnx.metrics.MultiMetric, clip_eps, ent_coef):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def train_minibatch(carry, minibatch):
        model, optimizer, metrics = carry
        (_, (actor_loss, critic_loss, entropy)), grad = grad_fn(model, minibatch, clip_eps, ent_coef)
        optimizer.update(model, grad)
        metrics.update(actor_loss=actor_loss, critic_loss=critic_loss, entropy=entropy)
        return model, optimizer, metrics

    train_minibatch((model, optimizer, metrics), minibatches)


def make_minibatches(batch, env_indices, envs_per_batch: int):
    env_ids = jnp.asarray(env_indices, dtype=jnp.int32).reshape(-1, envs_per_batch)

    def select_time_env(x):
        return jnp.swapaxes(jnp.take(x, env_ids, axis=1), 0, 1)

    return tuple(select_time_env(b) for b in batch)
