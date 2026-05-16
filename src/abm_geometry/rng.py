import subprocess

import jax
from jaxtyping import PRNGKeyArray


def make_key(seed: int) -> PRNGKeyArray:
    return jax.random.PRNGKey(seed)


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"
