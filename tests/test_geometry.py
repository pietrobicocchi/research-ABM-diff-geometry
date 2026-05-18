import jax
import jax.numpy as jnp
import numpy as np

from abm_geometry.config import Config
from abm_geometry.geometry.fim import estimate_noise_cov, fisher_information
from abm_geometry.geometry.spectrum import condition_number, eigendecomp, sloppy_modes
from abm_geometry.schelling.simulate import simulate_with_beta
from abm_geometry.schelling.state import init_world
from abm_geometry.statistics.summary import stats_array_fn

_CFG = Config(H=8, W=8, T=5, seed=11)
_KEY = jax.random.PRNGKey(0)


def _make_run_fn(cfg, tau_val, beta_val):
    """Returns a function: PRNGKey → Float[4] for noise cov estimation."""
    def run_fn(key):
        k_init, k_sim = jax.random.split(key)
        state = init_world(k_init, cfg).replace(
            tolerances=jnp.full((cfg.H, cfg.W), tau_val)
        )
        final = simulate_with_beta(k_sim, state, cfg, beta_val)
        return stats_array_fn(final, beta_val)
    return run_fn


def test_estimate_noise_cov_shape():
    run_fn = _make_run_fn(_CFG, 0.4, 5.0)
    Sigma = estimate_noise_cov(run_fn, _KEY, K=10)
    assert Sigma.shape == (4, 4)


def test_estimate_noise_cov_psd():
    """Noise covariance must be positive semi-definite."""
    run_fn = _make_run_fn(_CFG, 0.4, 5.0)
    Sigma = estimate_noise_cov(run_fn, _KEY, K=15)
    vals = jnp.linalg.eigvalsh(Sigma)
    assert jnp.all(vals >= -1e-6), f"Sigma not PSD: min eigenvalue={float(vals.min())}"


def test_fisher_information_shape():
    J = jnp.ones((4, 2))   # 4 stats, 2 params
    Sigma = jnp.eye(4)
    F = fisher_information(J, Sigma)
    assert F.shape == (2, 2)


def test_fisher_information_identity_sigma():
    """With Sigma=I, F = J^T J."""
    J = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    F = fisher_information(J, jnp.eye(4))
    expected = J.T @ J
    assert jnp.allclose(F, expected, atol=1e-5)


def test_eigendecomp_shape():
    F = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    vals, vecs = eigendecomp(F)
    assert vals.shape == (2,)
    assert vecs.shape == (2, 2)


def test_eigendecomp_descending():
    F = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    vals, _ = eigendecomp(F)
    assert float(vals[0]) >= float(vals[1])


def test_condition_number_diagonal():
    """Diagonal FIM with known ratio."""
    F = jnp.diag(jnp.array([100.0, 1.0]))
    vals, _ = eigendecomp(F)
    kappa = condition_number(vals)
    assert jnp.allclose(kappa, jnp.array(100.0), atol=1e-3)


def test_sloppy_modes_threshold():
    """With threshold=0.1, the small eigenvalue should be identified as sloppy."""
    F = jnp.diag(jnp.array([100.0, 5.0]))
    vals, vecs = eigendecomp(F)
    modes = sloppy_modes(vals, vecs, threshold=0.1)
    assert len(modes) == 1
    assert jnp.allclose(modes[0][0], jnp.array(5.0), atol=1e-3)
