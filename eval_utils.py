from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import imageio.v3 as iio


def load_checkpoint(model, checkpoint_dir, step=None):
    _, abstract_state = nnx.split(model)
    mngr = ocp.CheckpointManager(Path(checkpoint_dir).resolve())
    if step is None:
        step = mngr.latest_step()
    restored = mngr.restore(step, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(abstract_state),
    ))
    nnx.update(model, restored.model)
    mngr.close()
    return step


@nnx.jit
def greedy_action(model, obs, state):
    logits, _, next_state = model.step(obs, state)
    logits = logits.astype(jnp.float32)
    actions = jnp.argmax(logits, axis=-1)
    return actions, next_state


def save_video(frames, path, fps):
    video = np.stack(frames)
    iio.imwrite(path, video, fps=fps, codec="h264")
