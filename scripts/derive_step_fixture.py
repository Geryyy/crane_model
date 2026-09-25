#!/usr/bin/env python3
"""
Derive `test/step_response_fixture.json` from the HydraulicCalib campaign.

Offline, run by hand, and the only thing in this package that needs the bags.
They live outside the repo (`~/Documents/HydraulicCalib/bags`, 78 staircases,
read-only) and are never committed; what gets committed is the JSON this
writes, which is what `test/test_step_response.py` holds the C3 fit to.

    scripts/derive_step_fixture.py [--bags DIR] [--out PATH] [--survey]

**The response is normalised by the velocity the axis actually reached, not by
the command.** `/setpoints_compensated` does not carry a joint velocity: it is
`u_norm` in [-1, 1], downstream of Psi, and turning it back into rad/s needs
`calibration/c3/psi.py`'s identified splines -- load-dependent on the boom.
C3 has unit DC gain, so normalising the measured step by its own settled change
compares the *shape* -- dead time, lag, omega_n, zeta -- and drops Psi out of
the comparison entirely, which is right: Psi is assumed exact everywhere in
this package. It also makes the fixture amplitude-free, so the linear model can
be driven with a unit step.

One entry per planned axis, and each is the **mean over every clean edge in one
bag**, not a single step: the edges repeat the same level ladder, the encoder
velocity is differentiated and noisy, and averaging 4-8 of them at one pose is
what makes the late samples mean anything. The spread across those edges is
stored beside the mean and is where the test's tolerance comes from.

Edges are taken between two held staircase levels, never off rest: the
breakaway from zero carries stiction and a valve deadband, neither of which C3
models, so a step from rest would indict the model for a nonlinearity it never
claimed.

The bag is picked by lowest spread and never by how well the model does on it.

`m_ii` is `calibration/c3/rbd.Inertia`, i.e. CRBA on the bag's own
`/robot_description` with the telescope mimic projected -- the same call the fit
took its `m_ii_median` from, which it reproduces to the digit on `sa` and `ro`.
It is stored rather than recomputed in the test because the committed
`pzs100.urdf` is a different machine end: 21-30 % heavier on sw/ha/ka/sa and 3x
lighter on ro, so recomputing there would test the tool swap, not the actuator.
Pose is stored with it because `M_ii` is pose-dependent -- slewing's runs 836 to
23400 over the working range -- and a response compared at the wrong pose says
nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

#: The five planned axes, `symbolic.K_AXIS_KEYS`. `gr` is in the fit file but
#: not planned, and its Psi record predates the PZS100 rail anyway.
AXES = ("sw", "ha", "ka", "sa", "ro")

#: `fit_actuator_model.CAMPAIGN` -- one campaign glob per excited axis.
CAMPAIGN = {"sw": "M01v*", "ha": "M02v*", "ka": "M03v*", "sa": "M04v*", "ro": "M05v*"}

#: Column in `velocities_sp` (`rosbag_utils.AXIS_NAMES` order).
SETPOINT = {"sw": 0, "ha": 1, "ka": 2, "sa": 3, "ro": 4}

#: Canonical coordinate row, == `fit_actuator_model.BAG_JOINT`.
JOINT = {"sw": 0, "ha": 1, "ka": 2, "sa": 3, "ro": 6}

#: The URDF joint per canonical slot, as the *campaign's* description spells
#: them. Slot 7 is the timber grapple's jaw, not the PZS100 rail.
BAG_JOINTS = (
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta6_tip_joint",
    "theta7_tilt_joint",
    "theta8_rotator_joint",
    "theta10_outer_jaw_joint",
)

#: Where the response is read, seconds after the edge. Truncated per axis to
#: what that bag's held level actually covers. The first is the pinned 60 ms
#: transport delay: nothing may have happened yet.
GRID = (0.06, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.90, 1.20)

PRE = 40  # samples of held level required before the edge
MIN_HOLD = 90  # and after; shorter levels cannot settle
MIN_STEP = 0.02  # in u_norm, below which the edge is noise
QUIET = 1.0e-3  # every other axis' setpoint, so one axis moved alone
MIN_SNR = 6.0  # settled change over pre-step scatter
MIN_EDGES = 3  # fewer than this is one step, not an ensemble


def bag_reader(explicit: str | None):
    """
    Import `read_bag_full` and `Inertia` out of the `timber_crane_mujoco_py` tree.

    Imported rather than restated: `Inertia` is what the C3 fit itself weighed
    the bags with, down to the mimic projection and the cos/sin packing that a
    continuous joint needs, and a second copy of that would drift.
    """
    root = explicit or str(
        Path(__file__).resolve().parents[4] / "timber_crane_mujoco_py"
    )
    if root not in sys.path:
        sys.path.insert(0, root)
    from timber_crane_mujoco_py.calibration.c3.rbd import Inertia
    from timber_crane_mujoco_py.utils.rosbag_utils import read_bag_full

    return read_bag_full, Inertia


def edges(bag, axis: str) -> list[dict]:
    """Every level-to-level edge on `axis` with nothing else commanded."""
    command = bag["velocities_sp"][:, SETPOINT[axis]]
    rate = bag["velocities_js"][:, JOINT[axis]]
    others = np.delete(bag["velocities_sp"], SETPOINT[axis], axis=1)
    alone = np.all(np.abs(others) <= QUIET, axis=1)
    step = float(np.median(np.diff(bag["timestamps_js"])))

    found = []
    for n in range(PRE, len(command) - MIN_HOLD):
        if abs(command[n] - command[n - 1]) < MIN_STEP:
            continue
        if abs(command[n - 1]) < MIN_STEP:  # not off rest
            continue
        if np.ptp(command[n - PRE : n]) > 1e-9:
            continue
        end = n
        while end < len(command) - 1 and abs(command[end + 1] - command[n]) < 1e-9:
            end += 1
        hold = end - n
        if hold < MIN_HOLD or not np.all(alone[n - PRE : end]):
            continue
        tail = max(20, hold // 4)
        before = float(np.mean(rate[n - PRE : n]))
        settled = float(np.mean(rate[end - tail : end]))
        scatter = float(np.std(rate[n - PRE : n]))
        if abs(settled - before) < MIN_SNR * scatter:
            continue
        found.append(
            {
                "sample": n,
                "hold": hold,
                "dt_s": step,
                "before": before,
                "settled": settled,
                "u_norm": [float(command[n - 1]), float(command[n])],
            }
        )
    return found


def shapes(found: list[dict], rate) -> tuple[list[float], np.ndarray]:
    """
    Return the grid this ensemble supports, and one normalised row per edge.

    The grid stops before the window the settled value is averaged over, so no
    sample is read out of its own normaliser.
    """
    hold = min(e["hold"] for e in found)
    step = found[0]["dt_s"]
    times = [t for t in GRID if t <= (hold - max(20, hold // 4)) * step]
    rows = [
        [
            (rate[e["sample"] + round(t / step)] - e["before"])
            / (e["settled"] - e["before"])
            for t in times
        ]
        for e in found
    ]
    return times, np.asarray(rows, dtype=float)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bags", default=str(Path.home() / "Documents/HydraulicCalib/bags")
    )
    parser.add_argument(
        "--out", default=str(here.parent / "test/step_response_fixture.json")
    )
    parser.add_argument(
        "--mujoco-py", default=None, help="timber_crane_mujoco_py checkout"
    )
    parser.add_argument(
        "--survey", action="store_true", help="every bag, write nothing"
    )
    args = parser.parse_args()

    read_bag_full, Inertia = bag_reader(args.mujoco_py)
    inertia = None
    entries = {}
    for axis in AXES:
        best = None
        for path in sorted(glob.glob(f"{args.bags}/{CAMPAIGN[axis]}")):
            try:
                bag = read_bag_full(path)
            except Exception as error:  # M02v10 does not decode
                print(f"  skip {Path(path).name}: {type(error).__name__}")
                continue
            if inertia is None:
                inertia = Inertia(Path(path))
            found = edges(bag, axis)
            if len(found) < MIN_EDGES:
                continue
            times, rows = shapes(found, bag["velocities_js"][:, JOINT[axis]])
            samples = [e["sample"] for e in found]
            mass = inertia.diagonal(axis, bag["positions_js"][samples], stride=1)
            entry = {
                "bag": Path(path).name,
                "samples": samples,
                "dt_s": found[0]["dt_s"],
                "u_norm": [e["u_norm"] for e in found],
                "times_s": times,
                "response": [float(v) for v in rows.mean(axis=0)],
                "spread": [float(v) for v in rows.std(axis=0)],
                "m_ii": float(mass.mean()),
                "m_ii_range": [float(mass.min()), float(mass.max())],
                "q": [float(v) for v in bag["positions_js"][samples].mean(axis=0)],
            }
            if args.survey:
                print(
                    f"  {axis} {entry['bag']:8s} edges={len(found)} "
                    f"spread={np.mean(entry['spread']):.3f} m_ii={entry['m_ii']:9.1f}"
                )
            # Lowest spread wins, most edges breaks the tie. Never the residual
            # against the model: a fixture picked for agreeing is not evidence.
            key = (round(float(np.mean(entry["spread"])), 3), -len(found))
            if best is None or key < best[0]:
                best = (key, entry)
        if best is None:
            print(f"{axis}: no bag with {MIN_EDGES} clean edges; left out")
            continue
        entries[axis] = best[1]
        print(f"{axis}: {best[1]['bag']} {len(best[1]['samples'])} edges")

    if args.survey:
        return 0

    document = {
        "source": "HydraulicCalib staircases: /setpoints_compensated, /joint_states",
        "derived_by": "crane_model/scripts/derive_step_fixture.py",
        "response": "mean over the bag's edges of (dq(t) - dq_before)/(dq_settled - dq_before)",
        "u_norm": "the raw setpoint pair, in [-1, 1] and NOT rad/s -- provenance only",
        "q_joints": list(BAG_JOINTS),
        "q_note": "slot 7 is the campaign's timber grapple jaw, not the PZS100 rail",
        "m_ii_note": "rbd.Inertia on the bag's own /robot_description at q, mean over the edges",
        "t_zero": "the first sample carrying the new setpoint; the ZOH match makes that +-1 sample",
        "axes": entries,
    }
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
