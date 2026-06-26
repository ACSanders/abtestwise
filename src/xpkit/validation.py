"""Input validation helpers."""

from __future__ import annotations


def validate_count(name: str, value: int) -> None:
    """Validate a non-negative integer count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def validate_total(name: str, value: int) -> None:
    """Validate a positive integer total."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def validate_successes_le_total(
    successes_name: str,
    successes: int,
    total_name: str,
    total: int,
) -> None:
    """Validate that successes do not exceed total."""
    if successes > total:
        raise ValueError(
            f"{successes_name} ({successes}) cannot exceed {total_name} ({total})."
        )


def validate_prior(name: str, value: float) -> None:
    """Validate a positive Beta prior parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def validate_credible_interval(value: float) -> None:
    """Validate a credible interval level between 0 and 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"credible_interval must be a number, got {value!r}.")
    if not (0.0 < value < 1.0):
        raise ValueError(
            f"credible_interval must be strictly between 0 and 1, got {value}."
        )


def validate_n_simulations(value: int) -> None:
    """Validate a positive integer number of simulations."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"n_simulations must be an integer, got {value!r}.")
    if value <= 0:
        raise ValueError(f"n_simulations must be > 0, got {value}.")


def validate_seed(value: int | None) -> None:
    """Validate a random seed."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"seed must be None or an integer, got {value!r}.")
    if value < 0:
        raise ValueError(f"seed must be >= 0, got {value}.")