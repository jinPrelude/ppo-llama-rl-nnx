import unittest

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from transformer_nnx import TransformerBackbone, TransformerConfig, TransformerState
from ppo_core import CategoricalCritic
from train_ballet_symbolic import PPOTransformerBalletSymbolic, MODEL_DTYPE, PARAM_DTYPE

OBS_DIM = 8
NUM_ACTIONS = 4


def make_done(pattern):
    return jnp.asarray(pattern, dtype=jnp.float32)


class TransformerUnrollTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3

    def _assert_close(self, actual, expected):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)

    def _cases(self):
        return [
            {
                "name": "no_done_empty_prefix",
                "context_len": 4,
                "seq_len": 5,
                "valid_len": 0,
                "pos": 0,
                "done": [[0, 0, 0, 0, 0]] * self.batch_size,
            },
            {
                "name": "single_reset_mid",
                "context_len": 4,
                "seq_len": 5,
                "valid_len": 3,
                "pos": 5,
                "done": [[0, 0, 1, 0, 0]] * self.batch_size,
            },
            {
                "name": "multi_reset_full_prefix",
                "context_len": 4,
                "seq_len": 6,
                "valid_len": 4,
                "pos": 9,
                "done": [[0, 1, 0, 1, 0, 0]] * self.batch_size,
            },
            {
                "name": "context_len_one",
                "context_len": 1,
                "seq_len": 4,
                "valid_len": 1,
                "pos": 2,
                "done": [[0, 1, 0, 1]] * self.batch_size,
            },
        ]

    def test_backbone_parallel_unroll(self):
        """Test that backbone.unroll produces valid output shapes."""
        for case_idx, case in enumerate(self._cases()):
            with self.subTest(case=case["name"]):
                cfg = TransformerConfig(
                    hidden_dim=16,
                    n_layers=2,
                    n_heads=4,
                    context_len=case["context_len"],
                    dtype=jnp.float32,
                    param_dtype=jnp.float32,
                )
                backbone = TransformerBackbone(cfg, rngs=nnx.Rngs(case_idx))
                x_key = jax.random.PRNGKey(100 + case_idx)
                x_seq = jax.random.normal(x_key, (self.batch_size, case["seq_len"], cfg.hidden_dim), dtype=cfg.dtype)
                done_seq = make_done(case["done"])

                hidden = backbone.unroll(x_seq, done_seq)

                self.assertEqual(hidden.shape, (self.batch_size, case["seq_len"], cfg.hidden_dim))

    def test_ppo_transformer_ballet_unroll(self):
        """Test PPOTransformerBalletSymbolic.unroll returns correct shapes with CategoricalCritic."""
        for case_idx, case in enumerate(self._cases()):
            with self.subTest(case=case["name"]):
                cfg = TransformerConfig(
                    hidden_dim=16,
                    n_layers=2,
                    n_heads=4,
                    context_len=case["context_len"],
                    dtype=jnp.float32,
                    param_dtype=jnp.float32,
                )
                critic = CategoricalCritic(num_bins=11, value_min=0.0, value_max=1.0, sigma=0.1)
                model = PPOTransformerBalletSymbolic(
                    obs_dim=OBS_DIM,
                    num_actions=NUM_ACTIONS,
                    transformer_cfg=cfg,
                    critic=critic,
                    rngs=nnx.Rngs(1000 + case_idx),
                )
                obs_key = jax.random.PRNGKey(300 + case_idx)
                obs_seq = jax.random.normal(obs_key, (case["seq_len"], self.batch_size, OBS_DIM), dtype=jnp.float32)
                done_seq = make_done(case["done"]).T

                logits, critic_pred = model.unroll(obs_seq, done_seq)

                self.assertEqual(logits.shape, (case["seq_len"], self.batch_size, NUM_ACTIONS))
                self.assertEqual(critic_pred.logits.shape, (case["seq_len"], self.batch_size, critic.num_bins))
                self.assertEqual(critic_pred.value.shape, (case["seq_len"], self.batch_size))

    def test_ppo_transformer_ballet_step(self):
        """Test PPOTransformerBalletSymbolic.step returns correct shapes and advances state."""
        cfg = TransformerConfig(
            hidden_dim=16,
            n_layers=2,
            n_heads=4,
            context_len=4,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
        critic = CategoricalCritic(num_bins=11, value_min=0.0, value_max=1.0, sigma=0.1)
        model = PPOTransformerBalletSymbolic(
            obs_dim=OBS_DIM,
            num_actions=NUM_ACTIONS,
            transformer_cfg=cfg,
            critic=critic,
            rngs=nnx.Rngs(42),
        )
        obs = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, OBS_DIM), dtype=jnp.float32)
        state = model.init_state(self.batch_size)

        logits, critic_pred, next_state = model.step(obs, state)

        self.assertEqual(logits.shape, (self.batch_size, NUM_ACTIONS))
        self.assertEqual(critic_pred.value.shape, (self.batch_size,))
        np.testing.assert_array_equal(np.asarray(next_state.pos), np.ones(self.batch_size, dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
