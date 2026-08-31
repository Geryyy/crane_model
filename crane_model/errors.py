"""
Failure model for the Python crane model.

The C++ API returns `Result<T>`; here a failure is an exception carrying the
same `ErrorCode`, so call sites read as straight-line numeric code.
"""

from __future__ import annotations

import enum


class ErrorCode(enum.Enum):
    INVALID_ARGUMENT = "InvalidArgument"
    NON_FINITE_INPUT = "NonFiniteInput"
    INVALID_PAYLOAD = "InvalidPayload"
    INVALID_SCENE = "InvalidScene"
    MISSING_JOINT = "MissingJoint"
    UNSUPPORTED_TOOL = "UnsupportedTool"
    INVALID_ROBOT_DESCRIPTION = "InvalidRobotDescription"
    FRAME_UNAVAILABLE = "FrameUnavailable"
    SINGULAR_CONFIGURATION = "SingularConfiguration"
    BACKEND_UNAVAILABLE = "BackendUnavailable"
    COLLISION_BACKEND_FAILURE = "CollisionBackendFailure"


class CraneModelError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message


def _raise(code: ErrorCode, message: str) -> None:
    raise CraneModelError(code, message)
