"""Filter registry and discovery."""

from __future__ import annotations

from typing import Any

from pywolken.filters.base import Filter


class FilterRegistry:
    """Registry for filter classes, keyed by type name."""

    def __init__(self) -> None:
        self._filters: dict[str, type[Filter]] = {}

    def register(self, cls: type[Filter]) -> None:
        """Register a filter class."""
        self._filters[cls.type_name()] = cls

    def get(self, type_name: str, **options: Any) -> Filter:
        """Create a filter instance by type name."""
        if type_name not in self._filters:
            raise ValueError(
                f"Unknown filter '{type_name}'. "
                f"Available: {list(self._filters.keys())}"
            )
        return self._filters[type_name](**options)

    @property
    def available(self) -> list[str]:
        """List of registered filter type names."""
        return list(self._filters.keys())


# Global registry
filter_registry = FilterRegistry()

_registered = False


def _ensure_registered() -> None:
    """Lazy-register all built-in filters."""
    global _registered
    if _registered:
        return
    _registered = True

    # Import all built-in filters (they register themselves on import)
    from pywolken.filters import range as _range  # noqa: F401
    from pywolken.filters import crop as _crop  # noqa: F401
    from pywolken.filters import merge as _merge  # noqa: F401
    from pywolken.filters import decimation as _decimation  # noqa: F401
    from pywolken.filters import assign as _assign  # noqa: F401
    from pywolken.filters import expression as _expression  # noqa: F401
    from pywolken.filters import reprojection as _reprojection  # noqa: F401
    from pywolken.filters import ground as _ground  # noqa: F401
    from pywolken.filters import hag as _hag  # noqa: F401


def get_filter(type_name: str, **options: Any) -> Filter:
    """Get a filter instance by type name (convenience function)."""
    _ensure_registered()
    return filter_registry.get(type_name, **options)
