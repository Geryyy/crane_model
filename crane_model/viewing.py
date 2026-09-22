"""
What a viewer run needs beyond the plant: TCP overlays and a key gate.

Overlays go into the viewer's own `user_scn`, which survives a `sync` and is
drawn over -- never into the model, so nothing here can move what is simulated.
`MujocoPlant.positions_of` does the kinematics on its scratch data.
"""

from __future__ import annotations

import numpy as np

from .conventions import GENERALIZED_DOF, PASSIVE_INDICES, Frame
from .mujoco_plant import PLANNED_INDICES, TOOL_INDEX

#: The frame the overlays are drawn through, which is a body once MuJoCo has
#: compiled the URDF. `Frame` is the only place that link name is spelled.
TCP_BODY = Frame.TCP.value

#: GLFW's space bar; the viewer hands the callback raw key codes.
KEY_SPACE = 32

#: Points per curve. One capsule per segment costs a geom and a pybind call,
#: and the plan arrives with hundreds of samples nobody can tell apart.
MAX_POINTS = 96


def canonical_rows(planned, passive, q_tool: float) -> np.ndarray:
    """(N, 8) canonical configurations from the planned five and the passive pair."""
    planned = np.atleast_2d(np.asarray(planned, dtype=float))
    passive = np.atleast_2d(np.asarray(passive, dtype=float))
    rows = np.zeros((len(planned), GENERALIZED_DOF))
    rows[:, list(PLANNED_INDICES)] = planned
    rows[:, list(PASSIVE_INDICES)] = passive
    rows[:, TOOL_INDEX] = q_tool
    return rows


def _thin(rows: np.ndarray) -> np.ndarray:
    """At most `MAX_POINTS` rows, both endpoints kept."""
    if len(rows) <= MAX_POINTS:
        return rows
    return rows[np.linspace(0, len(rows) - 1, MAX_POINTS).round().astype(int)]


class Markers:
    """
    Named TCP curves in the viewer's user scene.

    One entry per name, redrawn whole on every change: `user_scn` is a flat geom
    list with no handle back into it, so there is nothing to edit in place.
    """

    def __init__(self, plant) -> None:
        import mujoco

        self._mj = mujoco
        self._plant = plant
        self._lines: dict[str, tuple] = {}

    def path(self, name: str, q_rows, rgba, width: float) -> None:
        """Draw the TCP curve of `q_rows` -- canonical eight per row -- as `name`."""
        rows = _thin(np.atleast_2d(np.asarray(q_rows, dtype=float)))
        self._lines[name] = (
            self._plant.positions_of(TCP_BODY, rows),
            np.asarray(rgba, dtype=np.float32),
            float(width),
        )
        self._draw()

    def drop(self, name: str) -> None:
        if self._lines.pop(name, None) is not None:
            self._draw()

    def _draw(self) -> None:
        viewer = self._plant.viewer
        if viewer is None or not viewer.is_running():
            return
        scene = viewer.user_scn
        capsule = self._mj.mjtGeom.mjGEOM_CAPSULE
        zeros, identity = np.zeros(3), np.eye(3).reshape(-1)
        # Under the lock: the render thread reads `user_scn` while this empties
        # and refills it one geom at a time, and a curve half written is what a
        # frame taken in between would draw.
        with viewer.lock():
            scene.ngeom = 0
            for points, rgba, width in self._lines.values():
                for tail, head in zip(points[:-1], points[1:]):
                    if scene.ngeom >= scene.maxgeom:
                        return
                    geom = scene.geoms[scene.ngeom]
                    self._mj.mjv_initGeom(geom, capsule, zeros, zeros, identity, rgba)
                    self._mj.mjv_connector(geom, capsule, width, tail, head)
                    scene.ngeom += 1


class SpaceGate:
    """The viewer's key callback and the wait it arms: space releases the next move."""

    def __init__(self) -> None:
        self.pressed = False

    def __call__(self, key: int) -> None:
        if int(key) == KEY_SPACE:
            self.pressed = True

    def wait(self, plant) -> bool:
        """
        Render until space, or until the window closes -- False means it closed.

        No viewer is not a closed one: headless, there is nobody to press space
        and the caller is meant to keep going.
        """
        self.pressed = False
        if plant.viewer is None:
            return True
        return plant.hold_viewer(until=lambda: self.pressed)
