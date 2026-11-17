"""Kill-switch / fail-closed tests for the sim harness (SIMULATED —
infrastructure validation, not user-behavior evidence).

Card-layer sims must fail closed when `cards_v4_1` is off, and the
SIFT_CARDS_V4_1=0 env kill-switch must be respected by the harness.
"""

import pytest

from sift import flags
from sift.cards.decay import DecayError
from sift.cards.service import LayerError
from sift.cards.share import ShareError
from sift.cards.vault import VaultError
from sift.sim.antiabuse import mint_sim_card, register_ready, good_post_for, _grant
from sift.sim.lifecycle import run_lifecycle_sim
from sift.sim.world import SimWorld


def test_sim_fails_closed_when_flag_off(sim_env, tmp_path):
    """With cards_v4_1 off, card-layer sim paths refuse (fail closed)."""
    flags.set_flag("cards_v4_1", False)
    try:
        with SimWorld(tmp_path / "w", seed=1, n_users=1) as world:
            user = world.users[0]
            # the composition root guard fires
            with pytest.raises(LayerError):
                world.layer.require_layer(user.handle)
            # lifecycle sim refuses (its require_layer fires)
            with pytest.raises(LayerError):
                run_lifecycle_sim(tmp_path / "life2", seed=1)
            # vault mutation refuses
            flags.set_flag("cards_v4_1", True)
            cid = mint_sim_card(world, user.handle, "ks")
            flags.set_flag("cards_v4_1", False)
            with pytest.raises(VaultError):
                world.layer.vault.discard(cid, user.handle)
            # share grant refuses
            register_ready(world, user.handle)
            good_post_for(world, cid)
            with pytest.raises(ShareError):
                world.layer.shares.request_share(
                    user.handle, cid, post_url="https://x.example/p",
                    device_id=user.device_id, ip=user.ip, asn=user.asn,
                    now=world.clock.now(),
                )
            # decay pass refuses
            with pytest.raises(DecayError):
                world.layer.decay.run_pass(now=world.clock.now())
    finally:
        flags.reset()


def test_env_kill_switch_respected(sim_env, monkeypatch, tmp_path):
    """SIFT_CARDS_V4_1=0 in the env disables the layer for the sim too."""
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    assert flags.cards_v4_1_enabled("sim-user-0000") is False
    with SimWorld(tmp_path / "w", seed=1, n_users=1) as world:
        with pytest.raises(LayerError):
            world.layer.require_layer(world.users[0].handle)
    monkeypatch.delenv("SIFT_CARDS_V4_1")
    flags.reset()
    assert flags.cards_v4_1_enabled("sim-user-0000") is True


def test_sim_world_requires_test_env(sim_env, monkeypatch, tmp_path):
    from sift.sim.world import assert_test_env
    monkeypatch.setenv("SIFT_ENV", "dev")
    with pytest.raises(RuntimeError, match="SIFT_ENV"):
        assert_test_env()
    monkeypatch.setenv("SIFT_ENV", "test")
    assert_test_env()  # no raise
