#!/usr/bin/env python3
"""
Inner-loop (JTC PI) bandwidth per planned axis against the sway frequency. Issue 173.

Offline, by hand. Writes `doc/inner_loop_bandwidth.md`.

    scripts/inner_loop_bandwidth.py [--out PATH]

Loop = disturbance-rejection loop of the JTC plugin (feed-forward excluded):
    u   = p e_pos + d e_vel = (p + d s) e_pos        (velocity_loop.py; i == 0)
    u  -> delay theta -> PT1 tau_v -> tau' = k (u_f - dq_i) -> body
theta = C3 dead time + half a 100 Hz sample (ZOH of the PI). u_clamp ignored (linear).

Body, linearised at the hanging equilibrium, other planned axes held (servoed):
- locked: pendulum rigid, M_ii, D_ii, K_ii (gravity stiffness of the axis itself)
- free:   rows/cols [i, tip, tilt] of M, dh/dq, dh/ddq -- the pendulum as the OCP has it
Sway = sqrt(eig(M_uu^-1 K_uu)), actuated locked.
Margins on the exact delay; closed-loop stability checked on a Pade(8) state space.
"""

from __future__ import annotations

import argparse
import itertools
import os

import casadi as ca
import crane_model.symbolic as cs
import numpy as np
from crane_model.conventions import Tool, canonical_joints
from crane_model.velocity_loop import load_velocity_loop

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DESCRIPTION = os.path.join(PKG, "test", "description", "pzs100.urdf")

DT_PI = 0.01
THETA = cs.K_ACTUATOR_FIT.dead_time_s + 0.5 * DT_PI
PM_MIN, GM_MIN = 45.0, 6.0
NAMES = ("slew", "boom", "arm", "telescope", "rotator")
# design load: "a concrete block of some 700-800 kg" (ledger 07, issue 067-080);
# point mass 1 m below the rotator mount (wiki/robot_model.md, r_z ~ +1 m)
PAYLOADS = (("0", 0.0), ("800 kg", 800.0))
TOOL_Q = 0.3
W = np.logspace(-2, 2.5, 6000)


def build():
    with open(DESCRIPTION, encoding="utf-8") as handle:
        model = cs.CraneSymbolicModel(handle.read(), Tool.PZS100)
    jac_x = ca.jacobian(model.bias, model.x)
    return ca.Function("mhk", [model.x, model.p], [model.mass, model.bias, jac_x])


def params(mass):
    p = np.zeros(cs.NP)
    p[cs.P_TOOL_POSITION] = TOOL_Q
    p[cs.P_PAYLOAD_MASS] = mass
    p[cs.P_PAYLOAD_COM + 2] = 1.0
    return p


def state(q_a, q_u):
    x = np.zeros(cs.NX_RIGID)
    x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF] = q_a
    x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF] = q_u
    return x


PASSIVE = list(cs.K_PASSIVE_ROWS)
PQ = [cs.X_PASSIVE_POSITION, cs.X_PASSIVE_POSITION + 1]


def equilibrium(f, q_a, p):
    """Hanging pose: h_u = 0 with K_uu > 0, seeded on a grid, then Newton."""
    best, seed = np.inf, None
    for a, b in itertools.product(np.linspace(-1.5, 1.5, 31), repeat=2):
        _, h, jx = (np.array(v) for v in f(state(q_a, [a, b]), p))
        k_uu = jx[np.ix_(PASSIVE, PQ)]
        if np.all(np.linalg.eigvalsh(0.5 * (k_uu + k_uu.T)) > 0):
            r = np.linalg.norm(h.ravel()[PASSIVE])
            if r < best:
                best, seed = r, np.array([a, b])
    q_u = seed
    for _ in range(50):
        _, h, jx = (np.array(v) for v in f(state(q_a, q_u), p))
        r = h.ravel()[PASSIVE]
        if np.linalg.norm(r) < 1e-8:
            break
        q_u = q_u - np.linalg.solve(jx[np.ix_(PASSIVE, PQ)], r)
    return q_u


_CACHE = {}


def body(f, q_a, p, axis):
    """(M, D, K) over [axis, tip, tilt] at the hanging pose, and the sway omegas."""
    key = (tuple(q_a), p[cs.P_PAYLOAD_MASS])
    if key not in _CACHE:
        q_u = equilibrium(f, q_a, p)
        _CACHE[key] = tuple(np.array(v) for v in f(state(q_a, q_u), p))
    m, _, jx = _CACHE[key]
    rows = [cs.K_PLANNED_ROWS[axis]] + PASSIVE
    qcols = [cs.X_PLANNED_POSITION + axis] + PQ
    dcols = [c + cs.K_PLANNED_DOF + cs.K_PASSIVE_DOF for c in qcols]
    mm, kk, dd = m[np.ix_(rows, rows)], jx[np.ix_(rows, qcols)], jx[np.ix_(rows, dcols)]
    sway = np.sqrt(
        np.sort(np.linalg.eigvals(np.linalg.solve(mm[1:, 1:], kk[1:, 1:])).real)
    )
    return mm, dd, kk, sway


def locked(mm, dd, kk):
    return mm[:1, :1], dd[:1, :1], kk[:1, :1]


def response(axis, mm, dd, kk, p_gain, d_gain, w=W):
    """Loop gain L(jw) = C * delay * PT1 * H, H: u_f -> q_i."""
    k, tau_v = cs.K_ACTUATOR_FIT.k[axis], cs.K_ACTUATOR_FIT.tau_v[axis]
    s = 1j * w
    e0 = np.zeros(len(mm))
    e0[0] = 1.0
    z = mm * s[:, None, None] ** 2 + dd * s[:, None, None] + kk
    g = np.linalg.solve(z, np.broadcast_to(e0, (len(s), len(e0)))[..., None])[:, 0, 0]
    h = k * g / (s * (1.0 + k * g))
    return (p_gain + d_gain * s) * np.exp(-s * THETA) / (tau_v * s + 1.0) * h


def pade(theta, n=8):
    """State space of the (n,n) Pade approximant of exp(-theta s)."""
    from math import factorial

    c = [
        factorial(2 * n - j)
        * factorial(n)
        / (factorial(2 * n) * factorial(j) * factorial(n - j))
        for j in range(n + 1)
    ]
    num = np.array([c[j] * (-theta) ** j for j in range(n + 1)])[::-1]
    den = np.array([c[j] * theta**j for j in range(n + 1)])[::-1]
    num, den = num / den[0], den / den[0]
    a = np.zeros((n, n))
    a[0, :] = -den[1:]
    a[1:, :-1] = np.eye(n - 1)
    b = np.zeros(n)
    b[0] = 1.0
    cc = num[1:] - num[0] * den[1:]
    return a, b, cc, num[0]


def stable(axis, mm, dd, kk, p_gain, d_gain):
    """Closed-loop eigenvalues on Pade(8): [pade | u_f? | tau | r | dr]."""
    k, tau_v = cs.K_ACTUATOR_FIT.k[axis], cs.K_ACTUATOR_FIT.tau_v[axis]
    ap, bp, cp, dp = pade(THETA)
    nr, npd = len(mm), len(ap)
    lag = tau_v > 0
    n = npd + int(lag) + 1 + 2 * nr
    iu, it, ir = npd, npd + int(lag), npd + int(lag) + 1
    iv = ir + nr
    a = np.zeros((n, n))
    # u = -(p q_i + d dq_i)
    u_row = np.zeros(n)
    u_row[ir], u_row[iv] = -p_gain, -d_gain
    a[:npd] += np.outer(bp, u_row)
    a[:npd, :npd] += ap
    delayed = np.zeros(n)
    delayed[:npd] = cp
    delayed += dp * u_row
    if lag:
        a[iu] = delayed / tau_v
        a[iu, iu] -= 1.0 / tau_v
        uf = np.zeros(n)
        uf[iu] = 1.0
    else:
        uf = delayed
    a[it] = k * uf
    a[it, iv] -= k
    a[ir:iv, iv:] = np.eye(nr)
    minv = np.linalg.inv(mm)
    a[iv:, ir:iv] = -minv @ kk
    a[iv:, iv:] = -minv @ dd
    a[iv:, it] = minv[:, 0]
    return np.max(np.linalg.eigvals(a).real) < 0.0


def margins(axis, mm, dd, kk, p_gain, d_gain):
    """(w_c, PM deg, GM dB, closed-loop -3 dB bandwidth, stable)."""
    L = response(axis, mm, dd, kk, p_gain, d_gain)
    mag, ph = np.abs(L), np.unwrap(np.angle(L))
    ph = ph - 2 * np.pi * np.round(ph[0] / (2 * np.pi) + 0.25)  # type 1: start near -90
    up = np.where(np.diff(np.sign(mag - 1.0)) != 0)[0]
    w_c = W[up[0]] if len(up) else np.nan
    pm = min((180.0 + np.degrees(ph[i]) for i in up), default=np.inf)
    cross = np.where(np.diff(np.sign(ph + np.pi)) != 0)[0]
    gm = min((-20 * np.log10(mag[i]) for i in cross), default=np.inf)
    T = np.abs(L / (1 + L))
    below = np.where(T < 1 / np.sqrt(2))[0]
    bw = W[below[0]] if len(below) else np.nan
    return w_c, pm, gm, bw, stable(axis, mm, dd, kk, p_gain, d_gain)


def grid(axis_range):
    q2 = np.linspace(0.0, 1.563, 4)
    q3 = np.linspace(-0.91, 1.3248, 4)
    q4 = np.linspace(0.0, 2.236, 3)
    q7 = (0.0, np.pi / 2)
    return [
        np.array([0.0, a, b, c, r]) for a, b, c, r in itertools.product(q2, q3, q4, q7)
    ]


def ok(axis, bodies, p_gain, d_gain, gm_min=-np.inf):
    worst = []
    for mm, dd, kk in bodies:
        w_c, pm, gm, bw, st = margins(axis, mm, dd, kk, p_gain, d_gain)
        if not st or not pm >= PM_MIN or not gm >= gm_min:
            return None
        worst.append((w_c, pm, gm, bw))
    return np.array(worst)


def best_p(axis, bodies, d_gain, gm_min=-np.inf):
    """Largest p (d fixed) meeting the margins at every pose; bisection on log p."""
    lo, hi = 1e-3, 100.0
    if ok(axis, bodies, lo, d_gain, gm_min) is None:
        return np.nan, None
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        if ok(axis, bodies, mid, d_gain, gm_min) is not None:
            lo = mid
        else:
            hi = mid
    return lo, ok(axis, bodies, lo, d_gain, gm_min)


def best_pd(axis, bodies):
    """(p, d), d on a grid, p maximal, maximising worst-pose BW at PM >= 45, GM >= 6 dB."""
    best = (np.nan, np.nan, 0.0, None)
    for d_gain in np.linspace(0.0, 1.5, 16):
        p_gain, res = best_p(axis, bodies, d_gain, GM_MIN)
        if res is not None and np.min(res[:, 3]) > best[2]:
            best = (p_gain, d_gain, np.min(res[:, 3]), res)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=os.path.join(PKG, "doc", "inner_loop_bandwidth.md")
    )
    parser.add_argument(
        "--quick", action="store_true", help="skip the (p, d) grid search"
    )
    options = parser.parse_args()

    f = build()
    gains, _ = load_velocity_loop()
    joints = canonical_joints()
    poses = grid(None)
    rows, ach = [], []
    for axis, name in enumerate(NAMES):
        g = gains[joints[cs.K_PLANNED_ROWS[axis]]]
        for tag, mass in PAYLOADS:
            all_free = []
            p = params(mass)
            data = []
            for q_a in poses:
                mm, dd, kk, sway = body(f, q_a, p, axis)
                data.append((q_a, mm, dd, kk, sway))
                all_free.append((mm, dd, kk))
            order = np.argsort([d[1][0, 0] for d in data])
            picks = {"min": order[0], "mid": order[len(order) // 2], "max": order[-1]}
            for which, idx in picks.items():
                q_a, mm, dd, kk, sway = data[idx]
                m_free = mm[0, 0] - mm[0, 1:] @ np.linalg.solve(mm[1:, 1:], mm[1:, 0])
                lk = margins(axis, *locked(mm, dd, kk), g.p, g.d)
                fr = margins(axis, mm, dd, kk, g.p, g.d)
                rows.append((name, tag, which, q_a, mm[0, 0], m_free, sway[0], lk, fr))
                print(
                    rows[-1][:3],
                    f"M={mm[0, 0]:.0f} sway={sway[0]:.2f}",
                    lk,
                    fr,
                    flush=True,
                )
            # achievable: worst case over the pose grid, free pendulum
            shipped = [margins(axis, *b, g.p, g.d) for b in all_free]
            bad = sum(1 for r in shipped if not r[4] or r[1] < PM_MIN)
            p_max, res = best_p(axis, all_free, g.d)
            pd = (
                (np.nan, np.nan, np.nan, None)
                if options.quick
                else best_pd(axis, all_free)
            )
            ach.append((name, tag, g, bad, p_max, res, pd))
            print("achievable", name, tag, p_max, pd[:3], flush=True)

    write(options.out, rows, ach, gains, joints)


def fmt(v, spec=".2f"):
    return "-" if v is None or not np.isfinite(v) else format(v, spec)


def write(path, rows, ach, gains, joints):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = [
        "# Inner-loop bandwidth vs sway (issue 173)",
        "",
        "Generated by `scripts/inner_loop_bandwidth.py`; do not hand-edit.",
        "",
        f"Loop: `u = (p + d s) e_pos` (JTC PI, feed-forward excluded, clamp ignored) -> "
        f"delay {THETA * 1e3:.0f} ms (C3 60 ms + half a 100 Hz ZOH sample) -> PT1 `tau_v` -> "
        "`tau' = k (u_f - dq)` -> body. Body linearised at the hanging equilibrium, other "
        "planned axes held; **locked** = pendulum rigid (M_ii), **free** = pendulum as the "
        "OCP models it (rows [i, tip, tilt] of M, dh/dq, dh/ddq). Payload 800 kg point mass "
        "1 m below the mount. Pose grid: slew 0, boom 0..1.563 (4), arm -0.91..1.325 (4), "
        "telescope 0..2.236 (3), rotator 0/pi/2 -- 96 poses; min/mid/max by locked M_ii. "
        "Units rad/s (w), kg m^2 or kg (M). `sway` = lowest pendulum mode, actuated locked. "
        "`BW` = -3 dB of `q_ref -> q` without feed-forward. GM `inf` = no -180 crossing.",
        "",
        "## Shipped gains",
        "",
        "| axis | p | d | payload | pose | M_ii | M_ii free-pend | w_sway | "
        "w_c lk | PM lk | GM lk dB | BW lk | w_c fr | PM fr | GM fr dB | BW fr | BW fr / w_sway |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, tag, which, q_a, m, m_free, sway, lk, fr in rows:
        g = gains[joints[cs.K_PLANNED_ROWS[NAMES.index(name)]]]
        pose = which + " (" + ",".join(f"{v:.2f}" for v in q_a[1:]) + ")"
        cells = [
            name,
            fmt(g.p),
            fmt(g.d, ".3f"),
            tag,
            pose,
            fmt(m, ".0f"),
            fmt(m_free, ".0f"),
            fmt(sway),
        ]
        for w_c, pm, gm, bw, st in (lk, fr):
            cells += [
                fmt(w_c),
                fmt(pm, ".0f") + ("" if st else " UNSTABLE"),
                fmt(gm, ".1f") if np.isfinite(gm) else "inf",
                fmt(bw),
            ]
        cells.append(fmt(fr[3] / sway))
        out.append("| " + " | ".join(cells) + " |")
    out += [
        "",
        "## Achievable (free pendulum, worst case over the 96-pose grid, per payload)",
        "",
        "`p_max`: largest `p` at shipped `d` keeping PM >= 45 deg and closed-loop stability at "
        "every point. `best (p, d)`: grid d in 0..1.5, p in 0.03..50, PM >= 45 and GM >= 6 dB "
        "everywhere, maximising the worst-point bandwidth. Columns give the worst point.",
        "",
        "| axis | payload | shipped p / d | poses PM<45 or unstable | p_max (d shipped) | max w_c | min PM | min GM dB | "
        "min BW | best (p, d) | min BW at best |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, tag, g, bad, p_max, res, pd in ach:
        head = [name, tag, f"{g.p:.2f} / {g.d:.3f}", f"{bad}/96"]
        if res is None:
            cells = head + ["-", "-", "-", "-", "-", "-"]
        else:
            cells = head + [
                fmt(p_max),
                fmt(np.max(res[:, 0])),
                fmt(np.min(res[:, 1]), ".0f"),
                fmt(np.min(res[:, 2]), ".1f"),
                fmt(np.min(res[:, 3])),
            ]
        cells += [f"{fmt(pd[0])}, {fmt(pd[1])}", fmt(pd[2])]
        out.append("| " + " | ".join(cells) + " |")
    out += ["", "## Conclusion", "", "(written by hand after the run)", ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
