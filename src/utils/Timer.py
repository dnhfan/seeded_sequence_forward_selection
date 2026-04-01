import functools
import time
from typing import Any, Callable


def _convert_seconds(elapsed_seconds: float, unit: str) -> float:
    """
    Convert elapsed time in seconds to the requested unit.
    Supported units: 's', 'ms'
    """
    if unit == "s":
        return elapsed_seconds
    if unit == "ms":
        return elapsed_seconds * 1000.0

    raise ValueError(f" Unsupported unit '{unit}'. Use 's' or 'ms'.")


def timer(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    unit: str = "ms",
    enabled: bool = True,
    logger: Callable[[str], None] = print,
) -> Callable[..., Any]:
    """
    A decorator that benchmarks the execution time of a function or method.

    Supports two usage styles:
        @timer
        @timer(name="...", unit="s", enabled=True, logger=print)

    Args:
        func: The function to decorate (None when used with kwargs).
        name: Custom label for the log output. Defaults to function's __qualname__.
        unit: Time unit for output. Options: "ms" (default), "s".
        enabled: If False, the decorator acts as a pass-through. Defaults to True.
        logger: Callable used for logging output. Defaults to print.

    Returns:
        The wrapped function's return value unchanged.
    """

    def decorator(target_func: Callable[..., Any]) -> Callable[..., Any]:

        @functools.wraps(target_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled:
                return target_func(*args, **kwargs)

            label = name if name else target_func.__qualname__
            start = time.perf_counter()
            try:
                result = target_func(*args, **kwargs)
                return result
            finally:
                elapsed_s = time.perf_counter() - start
                elapsed = _convert_seconds(elapsed_s, unit)
                logger(f"[Timer] {label} took {elapsed:.3f} {unit}")

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


class TimerContext:
    def __init__(
        self,
        name: str | None = None,
        unit: str = "ms",
        logger: Callable[[str], None] = print,
        enabled: bool = True,
    ) -> None:
        self.name = name or "Code Block"
        self.unit = unit
        self.logger = logger
        self.elapsed: float = 0.0
        self.enabled = enabled

    def __enter__(self) -> "TimerContext":
        if self.enabled:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.enabled:
            elapsed_s = time.perf_counter() - self.start
            self.elapsed = _convert_seconds(elapsed_s, self.unit)
            self.logger(f"[Timer] {self.name} took {self.elapsed:.3f} {self.unit}")
