import jax
import jax.numpy as jnp
import pytest

from abm_geometry.config import Config
from abm_geometry.schelling.state import init_world

CFG = Config(H=10, W=10, T=5, density=0.8, group_ratio=0.5, tau=0.4)


def test_init_world_occupancy_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert state.soft_occupancy.shape == (10, 10, 3)


def test_init_world_tolerances_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert state.tolerances.shape == (10, 10)


def test_init_world_sums_to_one():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sums = jnp.sum(state.soft_occupancy, axis=-1)
    assert jnp.allclose(sums, jnp.ones((10, 10)), atol=1e-5)


def test_init_world_tolerances_constant():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert jnp.allclose(state.tolerances, jnp.full((10, 10), 0.4))


def test_init_world_step_zero():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    assert int(state.step) == 0


from abm_geometry.schelling.neighbours import neighbour_type_fraction


def test_neighbour_fraction_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    frac = neighbour_type_fraction(state.soft_occupancy)
    assert frac.shape == (10, 10, 2)


def test_neighbour_fraction_sums_to_one():
    """frac_A + frac_B should sum to 1 everywhere (by definition of fraction)."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    frac = neighbour_type_fraction(state.soft_occupancy)
    total = frac[..., 0] + frac[..., 1]
    assert jnp.allclose(total, jnp.ones((10, 10)), atol=1e-5)


def test_neighbour_fraction_all_same_type():
    """Grid entirely filled with type A → frac_A = 1 everywhere."""
    # All type A, no empty
    occ = jnp.zeros((5, 5, 3)).at[..., 1].set(1.0)  # all A
    frac = neighbour_type_fraction(occ)
    assert jnp.allclose(frac[..., 0], jnp.ones((5, 5)), atol=1e-5)
    assert jnp.allclose(frac[..., 1], jnp.zeros((5, 5)), atol=1e-5)


from abm_geometry.schelling.satisfaction import soft_satisfaction


def test_satisfaction_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sat = soft_satisfaction(state.soft_occupancy, state.tolerances, CFG.beta)
    assert sat.shape == (10, 10)


def test_satisfaction_range():
    """Satisfaction values should be in (0, 1) since they are sigmoid outputs."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sat = soft_satisfaction(state.soft_occupancy, state.tolerances, CFG.beta)
    assert jnp.all(sat >= 0.0) and jnp.all(sat <= 1.0)


def test_satisfaction_zero_tolerance():
    """With τ=0, every agent is satisfied (any non-zero similar fraction ≥ 0)."""
    key = jax.random.PRNGKey(1)
    state = init_world(key, CFG)
    # Replace tolerances with 0
    state_zero_tau = state.replace(tolerances=jnp.zeros((10, 10)))
    sat = soft_satisfaction(state_zero_tau.soft_occupancy, state_zero_tau.tolerances, beta=10.0)
    # Most cells should be highly satisfied; none should be < 0.5 at high beta
    assert jnp.mean(sat) > 0.6


def test_satisfaction_high_tolerance():
    """With τ=1, no agent is ever fully satisfied."""
    key = jax.random.PRNGKey(1)
    state = init_world(key, CFG)
    state_high_tau = state.replace(tolerances=jnp.ones((10, 10)))
    sat = soft_satisfaction(state_high_tau.soft_occupancy, state_high_tau.tolerances, beta=10.0)
    assert jnp.mean(sat) < 0.4


from abm_geometry.schelling.move_rules import gumbel_softmax_step


def test_move_rule_output_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    new_occ = gumbel_softmax_step(
        state.soft_occupancy, state.tolerances, key, CFG.beta, CFG.tau_g
    )
    assert new_occ.shape == (10, 10, 3)


def test_move_rule_sums_to_one():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    new_occ = gumbel_softmax_step(
        state.soft_occupancy, state.tolerances, key, CFG.beta, CFG.tau_g
    )
    sums = jnp.sum(new_occ, axis=-1)
    assert jnp.allclose(sums, jnp.ones((10, 10)), atol=1e-5)


def test_move_rule_total_agent_mass_conserved():
    """Total mass of type A + type B should be conserved across a step."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    total_before = jnp.sum(state.soft_occupancy[..., 1:])
    new_occ = gumbel_softmax_step(
        state.soft_occupancy, state.tolerances, key, CFG.beta, CFG.tau_g
    )
    total_after = jnp.sum(new_occ[..., 1:])
    assert jnp.allclose(total_before, total_after, atol=1e-4)


def test_move_rule_non_negative():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    new_occ = gumbel_softmax_step(
        state.soft_occupancy, state.tolerances, key, CFG.beta, CFG.tau_g
    )
    assert jnp.all(new_occ >= -1e-6)


from abm_geometry.schelling.simulate import simulate


def test_simulate_output_shape():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    final = simulate(key, state, CFG)
    assert final.soft_occupancy.shape == (10, 10, 3)


def test_simulate_step_counter():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    final = simulate(key, state, CFG)
    assert int(final.step) == CFG.T


def test_simulate_mass_conservation():
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    final = simulate(key, state, CFG)
    sums = jnp.sum(final.soft_occupancy, axis=-1)
    assert jnp.allclose(sums, jnp.ones((10, 10)), atol=1e-4)


def test_simulate_jit():
    """Running jitted simulate twice with same inputs should give identical results."""
    key = jax.random.PRNGKey(0)
    state = init_world(key, CFG)
    sim_jit = jax.jit(simulate, static_argnums=(2,))
    out1 = sim_jit(key, state, CFG)
    out2 = sim_jit(key, state, CFG)
    assert jnp.allclose(out1.soft_occupancy, out2.soft_occupancy)


def test_simulate_vmap():
    """vmap over keys should produce a batch of independent simulations."""
    B = 4
    keys = jax.random.split(jax.random.PRNGKey(99), B)

    def run_one(key):
        state = init_world(key, CFG)
        return simulate(key, state, CFG)

    batch = jax.jit(jax.vmap(run_one))(keys)
    assert batch.soft_occupancy.shape == (B, 10, 10, 3)
