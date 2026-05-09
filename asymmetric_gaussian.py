"""Asymmetric normalized Gaussian distribution centered at 0."""

from __future__ import annotations

import math


def asymmetric_gaussian(x: float, g1: float, g2: float) -> float:
    """Return a continuous asymmetric Gaussian normalized to 1.

    The function is defined as:

    f(x) = A * exp(-x^2 / (2*g1^2)), x < 0
    f(x) = A * exp(-x^2 / (2*g2^2)), x >= 0

    where A = sqrt(2/pi) / (g1 + g2), which guarantees
    integral_{-inf}^{inf} f(x) dx = 1.
    """
    if g1 <= 0 or g2 <= 0:
        raise ValueError("g1 and g2 must be positive")

    norm = math.sqrt(2.0 / math.pi) / (g1 + g2)
    width = g1 if x < 0 else g2
    return norm * math.exp(-(x * x) / (2.0 * width * width))
