"""Strict scalar validation shared by the callable engine and rule configs."""
import math
from fractions import Fraction
from numbers import Real


def number(value, name, *, minimum=None, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number (not a string or boolean)")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside the permitted range [{minimum}, {maximum}]")
    return value


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def frame_rate(value):
    if isinstance(value, bool) or not isinstance(value, (Real, str, Fraction)):
        raise ValueError("target_fps must be a number or rational string")
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("target_fps must be finite and positive") from exc
    if not 0 < rate <= 240:
        raise ValueError("target_fps must be in (0, 240]")
    return rate


def dimension(value, name):
    number(value, name, positive=True)
    if int(value) != value:
        raise ValueError(f"{name} must be an integer pixel dimension")
    return int(value)
