"""Cosine-schedule resume tests.

`CosineAnnealingLR` carries `T_max` in its state_dict, so restoring a checkpoint
overwrote the horizon computed for the resumed run with the one from the run that
wrote it. The recursive cosine form is periodic, so stepping past `T_max` sends
the learning rate back up toward `base_lr` instead of holding at the floor --
and the trainer's own resume error tells the user to raise `num_epochs`, which is
exactly the path that triggers it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _sched(t_max, base_lr=0.003):
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=base_lr)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-5)


def test_cosine_is_periodic_past_its_horizon():
    """Documents the hazard the fix guards against."""
    opt, sch = _sched(100)
    for _ in range(100):
        sch.step()
    at_horizon = opt.param_groups[0]["lr"]
    for _ in range(60):
        sch.step()
    past_horizon = opt.param_groups[0]["lr"]
    assert at_horizon < 1e-4, f"expected the floor at T_max, got {at_horizon}"
    assert past_horizon > 100 * at_horizon, (
        "cosine should climb back up past T_max; if this no longer holds, the "
        "resume guard is protecting against nothing"
    )


def test_loading_a_checkpoint_overwrites_the_horizon():
    """The raw behaviour, without the guard."""
    _, old = _sched(100)
    for _ in range(50):
        old.step()
    state = old.state_dict()
    assert state["T_max"] == 100

    _, new = _sched(250)
    assert new.T_max == 250
    new.load_state_dict(state)
    assert new.T_max == 100, "load_state_dict is expected to clobber T_max"


def test_guard_restores_this_runs_horizon():
    """What `_restore_training_state` now does: keep the new horizon, take the rest."""
    _, old = _sched(100)
    for _ in range(50):
        old.step()
    state = old.state_dict()

    _, new = _sched(250)
    horizon = new.T_max
    new.load_state_dict(state)
    new.T_max = horizon

    assert new.T_max == 250
    assert new.last_epoch == 50, "step count must still come from the checkpoint"


def test_extended_run_stays_on_a_descending_schedule():
    """The end-to-end property: extend a finished run, LR must not climb back."""
    _, old = _sched(100)
    for _ in range(100):
        old.step()
    state = old.state_dict()

    opt, new = _sched(250)
    horizon = new.T_max
    new.load_state_dict(state)
    new.T_max = horizon

    lrs = []
    for _ in range(60):
        new.step()
        lrs.append(opt.param_groups[0]["lr"])

    assert all(
        b <= a + 1e-12 for a, b in zip(lrs, lrs[1:])
    ), "learning rate rose while extending a run; the horizon guard failed"
