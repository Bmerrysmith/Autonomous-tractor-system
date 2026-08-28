"""EMA warmup tests.

The EMA weights are what get scored and what get saved, so a cold start is not
cosmetic. With a fixed decay of 0.999 and 225 optimiser steps per epoch,
0.999 ** 225 = 0.798 -- the epoch-1 average is 80% random initialisation. The
retained 2026-07-30 run shows this directly: val_ema/cls_loss at epoch 1 is
4.513, and the analytically untrained value for prior_prob 0.01 is
(1 - 0.01)^2 * -ln(0.01) = 4.5135.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from agrinav.models.weeddet_v6b import ModelEMA


class _Tiny(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.w = nn.Parameter(torch.full((4,), float(value)))

    def forward(self, x):  # pragma: no cover - never called
        return x * self.w


def _drive(ema, steps, target=1.0):
    """Hold the live model at `target` and run `steps` EMA updates."""
    live = _Tiny(target)
    for _ in range(steps):
        ema.update(live)
    return ema.ema.w.detach().clone()


def test_untrained_value_matches_the_retained_run():
    """Anchors the motivating measurement so it cannot drift silently."""
    assert math.isclose((1 - 0.01) ** 2 * -math.log(0.01), 4.5135, abs_tol=1e-3)


def test_fixed_decay_is_still_mostly_initialisation_after_one_epoch():
    """Reproduces the defect. 225 steps at 0.999 leaves ~80% of the init."""
    ema = ModelEMA(_Tiny(0.0), decay=0.999, ramp=False)
    w = _drive(ema, 225, target=1.0)
    # Weight on the initialisation is decay**steps.
    assert math.isclose(float(w[0]), 1.0 - 0.999 ** 225, rel_tol=1e-3)
    assert float(w[0]) < 0.25, "expected the epoch-1 average to be mostly init"


def test_ramp_tracks_the_live_weights_early():
    """The fix: the same 225 steps should land close to the live model."""
    ema = ModelEMA(_Tiny(0.0), decay=0.999, ramp=True, ramp_tau=2000.0)
    w = _drive(ema, 225, target=1.0)
    assert float(w[0]) > 0.90, (
        f"ramped EMA still at {float(w[0]):.3f} after one epoch; the warmup is "
        "not doing its job")


def test_ramp_converges_to_the_configured_decay():
    ema = ModelEMA(_Tiny(0.0), decay=0.999, ramp=True, ramp_tau=100.0)
    ema.updates = 100_000
    assert math.isclose(ema.current_decay(), 0.999, rel_tol=1e-6)


def test_ramp_disabled_reproduces_the_old_constant_decay():
    ema = ModelEMA(_Tiny(0.0), decay=0.999, ramp=False)
    ema.updates = 1
    assert ema.current_decay() == 0.999
    ema.updates = 10_000
    assert ema.current_decay() == 0.999


def test_ramp_state_round_trips_through_a_checkpoint():
    """A resumed run must not restart the warmup."""
    ema = ModelEMA(_Tiny(0.0), decay=0.999, ramp=True, ramp_tau=2000.0)
    _drive(ema, 500)
    state = ema.state_dict()
    assert state["updates"] == 500

    fresh = ModelEMA(_Tiny(0.0), decay=0.999, ramp=True, ramp_tau=2000.0)
    assert fresh.updates == 0
    fresh.load_state_dict(state)
    assert fresh.updates == 500
    assert math.isclose(fresh.current_decay(), ema.current_decay(), rel_tol=1e-9)


def test_load_state_dict_tolerates_a_pre_ramp_checkpoint():
    """Older checkpoints carry no ema_state; that must not raise."""
    ema = ModelEMA(_Tiny(0.0), decay=0.999)
    ema.load_state_dict(None)
    ema.load_state_dict({})
    assert ema.updates == 0


def test_non_float_buffers_are_copied_not_averaged():
    """Integer buffers such as num_batches_tracked must be assigned outright."""
    class _WithBuf(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.w = nn.Parameter(torch.full((2,), float(v)))
            self.register_buffer("n", torch.tensor(int(v)))

    ema = ModelEMA(_WithBuf(0), decay=0.999, ramp=True)
    live = _WithBuf(7)
    ema.update(live)
    assert int(ema.ema.n) == 7
