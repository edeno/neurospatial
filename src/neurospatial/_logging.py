"""Structured logging infrastructure for neurospatial.

This module provides a centralized logging system for the neurospatial package.
By default, logging is disabled (NullHandler), but users can enable it by
configuring the root logger.

Examples
--------
Enable logging to console::

    import logging

    logging.basicConfig(level=logging.INFO)

    from neurospatial import Environment

    env = Environment.from_samples(positions, bin_size=2.0)
    # INFO:neurospatial:Environment created: 245 bins, 2D

Enable logging with more detail::

    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

Filter to only neurospatial logs::

    import logging

    neurospatial_logger = logging.getLogger("neurospatial")
    neurospatial_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    neurospatial_logger.addHandler(handler)
"""

import logging

# Create package logger
logger = logging.getLogger("neurospatial")

# Add NullHandler to prevent "No handler found" warnings
# Users must explicitly enable logging
logger.addHandler(logging.NullHandler())


def log_environment_created(
    env_type: str, n_bins: int, n_dims: int, env_name: str | None = None
) -> None:
    """Log environment creation.

    Parameters
    ----------
    env_type : str
        Type of environment (layout type tag).
    n_bins : int
        Number of bins in the environment.
    n_dims : int
        Number of spatial dimensions.
    env_name : str or None
        Optional name of the environment.
    """
    name_str = f" '{env_name}'" if env_name else ""
    logger.info(
        "Environment created%s: %d bins, %dD",
        name_str,
        n_bins,
        n_dims,
        extra={
            "type": env_type,
            "n_bins": n_bins,
            "n_dims": n_dims,
            "env_name": env_name,
        },
    )


def log_environment_combine(n_subenvs: int, total_bins: int, n_bridges: int) -> None:
    """Log composite environment construction.

    Parameters
    ----------
    n_subenvs : int
        Number of sub-environments merged.
    total_bins : int
        Total number of bins in composite.
    n_bridges : int
        Number of bridge edges created.
    """
    logger.info(
        "Environment.combine created: %d sub-envs, %d total bins, %d bridges",
        n_subenvs,
        total_bins,
        n_bridges,
        extra={
            "n_subenvs": n_subenvs,
            "total_bins": total_bins,
            "n_bridges": n_bridges,
        },
    )
