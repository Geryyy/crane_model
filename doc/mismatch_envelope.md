# Mismatch envelope for the robust MPC sweep (issue 174, step 1)

The plant-side deviations the offline chain is run against, each band sourced from data.
Multipliers are plant/shipped; the solver always keeps `config/c3_full_model.json`.
Axis order everywhere: slew (sw), boom (ha), arm (ka), telescope (sa), rotator (ro).

Data: `~/Documents/HydraulicCalib/` (78 staircase bags, one axis per bag, timber grapple,
no payload; `M02v10` does not decode). Fit code: `timber_crane_mujoco_py/calibration/c3/`
(`fit_full.py`, `fit_c3.py`, `decrement.py`, `steps/step0_command.py`); its README
("c3 README") holds the published spreads. Split is `i % 3` (`fit_full.py:385-386`).

**Refit 2026-09-28 (read-only, throwaway script, nothing committed).** The train/test spread
per axis was recorded only for slew, so `fit_full.prepare`/`axis_fit` were rerun with
`DEAD_COMMON` = 60 ms pinned (`fit_full.py:132`) on train, on test and on every bag
alone (k held where `K_HELD` holds it, `fit_full.py:96`), plus free-k on each split.
The plateau gain `qdot / Psi^-1(cmd)` was measured per bag and per sign with
`step0_command.plateaus`. These rows are marked **R**.

## Envelope

`nom` = shipped. `low`/`high` = sweep corners. Per-bag spreads quote the IQR; outliers
named in the source column are left out.

### C3 `k` (multiplier)

| axis | nom | low | high | source |
|---|---|---|---|---|
| sw | 3.191e5 (free fit) | 0.63 | 1.25 | **R** test split 0.74, per-bag IQR 0.63-0.98 (M01v20/21/31/32/33 degenerate); decrement 1.20 Hz -> 0.76 (c3 README:203); fit_c3 pressure 3.995e5 -> 1.25, pendulum-contaminated (README:310) |
| ha | 1.834e6 (held, fit_c3) | 0.70 | 1.50 | H1..H2 1.83-2.74e6 (`c3_actuator_model.json` `k_bracket_h1_h2`); lowering 1.31e6 = 0.71, lifting 0.89 (README:422); **R** free-k 0.92 / 1.06. Band sweep 0.2-1 Hz reaches 0.52 |
| ka | 5.78e5 (held) | 0.50 | 3.0 | **widened, 3 train / 2 test bags.** H2 x1.81; band sweep x16, 0.17-2.7 (README:396); **R** free-k 4.49 (train) / 3.09 (test); held-out R^2 -0.32 on fit_c3 |
| sa | 3.5e6 (geometric prior, not a fit) | 0.06 | 1.5 | not identified: velocity cost flat over 4e4..5.3e6 (`fit_full.py` K_HELD comment). **R** free-k 0.07 (train) / 0.05 (test). Decrement 2.18 Hz [1.74, 2.56] -> 0.09 [0.06, 0.13] at M 1690 (issue 150 notes) |
| ro | 7296 (held) | 0.35 | 1.72 | **widened, 2 train / 2 test bags.** H2 x1.15; band sweep 0.81-1.72; **R** free-k 0.49 (train) / 0.35 (test) |

### C3 `d` (multiplier)

| axis | nom | low | high | source |
|---|---|---|---|---|
| sw | 7520 | 0.84 | 2.2 | `d_within_5pc` 0.84-1.06 (`c3_full_model.json` sw); **R** test 1.61, per-bag IQR 1.01-2.15; decrement d 1.12e4 = 1.49 |
| ha | 1.656e5 | 0.40 | 2.3 | `d_within_5pc` 0.60-1.52; **R** per-bag 0.38-2.27, pose-split: the five light-pose bags (M_ii ~5000 vs 28500) want 0.38-0.55 at similar zeta, so d is not pose-invariant |
| ka | 4.70e4 | 0.70 | 1.5 | **widened.** `d_within_5pc` 0.85-1.34; **R** per-bag 0.95-1.15, test 1.14. Free-k fits pair (k 4.49, d 3.02) and (k 3.09, d 1.83), only as a pair |
| sa | 1.339e5 | 0.12 | 2.4 | `d_within_5pc` 0.12-2.36, "NOT identified"; **R** per-bag IQR 1.10-1.46 (max 2.07), free-k pairs (k 0.05-0.07, d 0.12-0.14); issue 152: exact-discretisation equivalent ~2.0 |
| ro | 483.7 | 0.13 | 3.3 | **widened.** `d_within_5pc` 0.13-3.28; **R** per-bag 0.75-2.3 (M05v9 5.9, NRMSE 0.55) |

`k` and `d` are one identification (`mismatch.py:86-95`). The paired corners the data
supports: sa (0.06, 0.13), ka (3.0, 2.4), ro (0.4, 0.7) from the free-k fits; elsewhere cross
the two columns.

### C3 `tau_v` (s, dead time pinned at 60 ms; only the sum is identified)

| axis | nom | low | high | source |
|---|---|---|---|---|
| sw | 0.100 | 0.025 | 0.15 | **R** per-bag IQR 0.025-0.088, test 0.050; `c3_dead_sweep.json` 0-0.15 over 0-200 ms |
| ha | 0.025 | 0.0 | 0.075 | **R** per-bag IQR 0-0.05 (M02v11 0.2) |
| ka | 0.0 | 0.0 | 0.05 | **R** 0 on every bag and split except free-k test 0.05 |
| sa | 0.050 | 0.0 | 0.075 | **R** per-bag 0-0.075 |
| ro | 0.125 | 0.10 | 0.25 | **R** per-bag 0.10-0.25, test 0.10 |

fit_c3 reads the same ridge from the pressure side at 200/150/125/-/75 ms (README:310): compare
sums, not lags.

### Dead time (s, common to all axes)

| axis | nom | low | high | source |
|---|---|---|---|---|
| all | 0.060 | 0.040 | 0.080 | held-out cost flat 40-80 ms, frozen-(k,d) argmin 60 on ha/ka/sa (`c3_dead_sweep*.json`, c3 README:124-131) |
| sw | 0.060 | 0.040 | 0.120 | free sweep argmin 120 ms (absorbs into k/d); valve echo bag median 90 ms, max 270 (`c3_transport_delay.json`) |
| ha | 0.060 | 0.020 | 0.080 | free sweep argmin 20 ms; echo median 60 ms, max 180 |
| ka | 0.060 | 0.040 | 0.080 | "arm fit 0" was the pre-pin per-axis grid read, a ridge artifact; the frozen sweep is a clean parabola at 60 ms; echo 60-90 ms |
| sa | 0.060 | 0.060 | 0.25 | breakaway: the telescope does not move for ~250 ms off rest, load-holding valve (issue 164 notes, `test_step_response.py` fixture); echo median 95 ms |
| ro | 0.060 | 0.060 | 0.150 | free argmin 150 / frozen 100 ms; echo median 110 ms |

Transport-delay echo = `i_d` vs `i` cross-correlation, a bus echo (hydraulics §7.2), so it
bounds the bus round trip, not the valve.

### Psi gain `g` = measured qdot / requested u, per sign

| axis | nom | + low | + high | - low | - high | source |
|---|---|---|---|---|---|---|
| sw | 1 | 0.98 | 1.02 | 0.98 | 1.02 | **R** + pooled 0.997 IQR 0.996-1.007; - 0.998 IQR 0.995-1.007 (18 bags; one bag 0.72) |
| ha | 1 | 0.95 | 1.15 | 0.93 | 1.05 | **R** 2-D surface as the machine runs it: + 1.038 IQR 1.02-1.14, - 0.974 IQR 0.95-1.03. The 1-D fallback (sim) would give - 1.26 IQR 1.23-1.47; surface load term up to 66 % of |qdot| boom-down (hydraulics §6.3, :339); issue 151 domain mismatch 3.4 % |
| ka | 1 | 0.95 | 1.15 | 1.00 | 1.20 | **widened, 2 bags with plateaus.** **R** + 1.03-1.06, - 1.08-1.15 |
| sa | 1 | 1.00 | 1.25 | 0.90 | 1.00 | **R** + 1.121 IQR 1.09-1.17, - 0.940 IQR 0.93-0.97 |
| ro | 1 | 1.05 | 1.20 | 1.05 | 1.20 | **widened, 4 bags.** **R** + 1.136, - 1.129 (two bags to 1.4-2.2) |

Pooled over signs these reproduce step0's 0.998/1.011/1.047/1.010/1.132 (c3 README:349-350).
Optimistic on ha: Psi's boom surface was fitted on the same 2026-06-26/07-16 sessions (§6.4).
Psi was off in the campaign, so this is Psi's residual error, not a closed-loop measurement.

### Payload and inertia

| quantity | nom | low | high | source |
|---|---|---|---|---|
| payload mass | 0 kg | 0 | 800 kg | block 777.6 kg (`concrete_block_behavior_tree/models/concrete_block/model.sdf:5`, 0.9x0.6x0.6 m, 2400 kg/m^3); design load 700-800 kg (`scripts/inner_loop_bandwidth.py:41`) |
| payload CoM (K8) | 0 0 1 m | - | - | point mass 1 m below the mount, as `inner_loop_bandwidth.py:59`; no measured offset |
| slew M_ii | pose | 851 | 69 419 (locked, empty); 159 590 at 800 kg | `doc/inner_loop_bandwidth.md` |

The OCP's payload parameter: `P_PAYLOAD_MASS` = the block's mass, `P_PAYLOAD_COM` = (0, 0, 1),
inertia 0 (point mass, as the node binds `crane_msgs/Payload`). Corners: known load (OCP 800,
plant 800), unknown load (OCP 0, plant 800) and a stale estimate (OCP 800, plant 0). No payload
estimator error has been measured, so there is no band between those corners. The bags carry
no payload at all, so every C3 number above is empty-hook.

Inertia is set by the pose, not a flag: pick moves through the heavy poses (boom/arm/telescope
out) or the 173 margin finding is not exercised.

### Timing

| quantity | nom | low | high | source |
|---|---|---|---|---|
| tick -> JTC receipt | 0.030 s | 0 | 0.060 | issue 172 body: 30 ms median sim, solve 10 ms median / 44 ms max. Above 60 ms (`solve_budget`) the cycle is a failed solve and the previous horizon shifts |
| late tail | 0 | - | 3-5 % of cycles at 61-99 ms | issue 172 notes (Gazebo, quiet host) |
| /joint_states age | 0 | 0 | 0.010 | 172 body item 2; since 172 the predictor rolls from the stamp |
| JTC period | 0.01 | - | - | configure refuses a mismatch (172, crane_mpc `74f5c2a`). No band |

No hardware latency or MPC-to-JTC clock offset has been measured.

### One-cycle skill (sanity)

`modelfit.py`, one-cycle velocity RMS, 5 bags per campaign (issue 169 notes): skill with the
carried force sw -2.3, ha -14.3, ka -37.9, sa -0.9, ro -20.2. With the measured force seeded:
0.07, -0.93, -3.66, -1.35, -0.17. So the one-cycle error is dominated by the force-state estimate.
A k/d band cannot move it that far, and it does not bound k/d. Issue 164's step fixture, the
worst |model - machine| on a normalised step: ro 0.09, sw 0.23, ha 0.23, ka 0.40, sa 0.79. It
ranks the axes the same way as the band widths above, with ka and sa widest.

## What the chain can set

`wire_chain.py` is the offline chain (the others were retired in crane_mpc issue 181);
its flags live in `scripts/trials/harness.py`.

| band | flag | notes |
|---|---|---|
| k | `--k-scale S S S S S` | must come with `--d-scale`; pass `--d-scale 1 1 1 1 1` for k alone |
| d | `--d-scale S S S S S` (+ `--k-scale 1 1 1 1 1`) | scales MuJoCo `dof_damping`, one constant per axis |
| Psi gain | `--psi-gain-positive G*5`, `--psi-gain-negative G*5` | `PsiGain.apply`, per sign of u (`u >= 0` is positive) |
| dead time | `--dead-time D` | in wire_chain this is **plant only**: the predictor is `Cycle(..., sensor_to_valve_delay)`. It must be a multiple of `--timestep` (0.5 ms). |
| payload | `--payload-mass M --payload-com 0 0 1` (OCP), `--plant-payload M` (MuJoCo) | `--plant-payload` and `carry_payload` are in crane_mpc's working tree, **uncommitted** (another session's work) |
| latency | `--latency L` (wire_chain) | constant delivery delay; the solve budget is off (10 s) |
| PI | `--pi-scale s` (wire_chain) | lever, not a band |
| MPC levers | `--set key=value` (working tree) | e.g. `weights.du`, `dq_a_feedback` |

## Gaps: bands the chain cannot express yet

1. **Per-axis `tau_v` / lag split.** `--lag-shift` is refused on the shipped fit (ka at
   `tau_v = 0`, `mismatch.py:108-120`) and is common to all axes anyway. Only the total delay
   can move, through `--dead-time`, and only on every axis at once. The per-axis dead-time rows
   (sa 250 ms, ro 150 ms) cannot be applied to one axis alone.
2. **Late-cycle tail.** `--latency` is one constant and the budget is off, so the 3-5 % dropped
   or shifted cycles are not drawn. `/joint_states` age is not separate from delivery, because
   wire_chain measures exactly at the tick. Fold it into `--latency` (+10 ms) at most.
4. **Direction-dependent `k`** (boom 20 % softer lowering) and **pose-dependent `d`** (the ha
   light-pose bags) are out of reach, because `k`/`d` are one scalar per axis.
5. **Unmodelled structure:** the sa load-holding-valve breakaway (~250 ms, a stiction block
   rather than a delay), telescope wear-pad Coulomb friction, and slew's second mode (1.1 vs
   1.38 Hz, the pendulum through the bearing).
6. **Payload estimate error and CoM offset:** no data, corners only.
7. **Per-axis corners** are supported by `sweep_robust.py` (`--band name=[lo x5],[hi x5]`);
   direction-asymmetric Psi gains are not (one value for both signs per axis).
8. **Hardware timing:** no machine latency or clock offset. The timing rows are Gazebo numbers.
