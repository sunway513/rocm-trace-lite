"""Adapter registry — @register_adapter decorator + AdapterRegistry class.

Usage:
    from profiler_perf_bench.adapters.registry import global_registry, register_adapter

    @global_registry.register
    class MyAdapter(ProfilerAdapter):
        name = "my_adapter"
        ...
"""

from typing import Dict, List, Type
from .base import ProfilerAdapter


class AdapterRegistry:
    """Registry for ProfilerAdapter subclasses."""

    def __init__(self):
        self._adapters: Dict[str, Type[ProfilerAdapter]] = {}

    def register(self, cls: Type[ProfilerAdapter]) -> Type[ProfilerAdapter]:
        """Decorator: register an adapter class by its name attribute.

        Raises ValueError on duplicate names (per spec §8 guardrail).
        """
        name = getattr(cls, "name", None)
        if not name:
            raise ValueError(f"Adapter class {cls.__name__} must define a 'name' attribute")
        if name in self._adapters:
            raise ValueError(
                f"Adapter name collision: '{name}' is already registered "
                f"(existing: {self._adapters[name].__name__}, new: {cls.__name__}). "
                "Use a different name."
            )
        self._adapters[name] = cls
        return cls

    def get(self, name: str) -> Type[ProfilerAdapter]:
        """Return adapter class by name. Raises KeyError if not found."""
        if name not in self._adapters:
            raise KeyError(f"No adapter registered with name '{name}'. "
                           f"Available: {self.list_names()}")
        return self._adapters[name]

    def list_names(self) -> List[str]:
        """Return sorted list of registered adapter names."""
        return sorted(self._adapters.keys())

    def enumerate(self) -> List[Type[ProfilerAdapter]]:
        """Return all registered adapter classes in sorted name order."""
        return [self._adapters[n] for n in self.list_names()]

    def __contains__(self, name: str) -> bool:
        return name in self._adapters


# Module-level global registry — all adapters self-register here
global_registry = AdapterRegistry()

# Convenience alias for use in adapter modules
register_adapter = global_registry.register
