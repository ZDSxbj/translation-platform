"""Translation engine factory."""

from app.engines.base_engine import BaseEngine
from app.engines.his2trans.engine import His2TransEngine


_engine_registry: dict[str, type[BaseEngine]] = {
    "his2trans": His2TransEngine,
}


def get_engine(name: str) -> BaseEngine:
    """Get a translation engine instance by name."""
    engine_cls = _engine_registry.get(name)
    if engine_cls is None:
        raise ValueError(f"Unknown engine: {name}. Available: {list(_engine_registry.keys())}")
    return engine_cls()


def register_engine(name: str, engine_cls: type[BaseEngine]):
    """Register a new engine (for extensibility)."""
    _engine_registry[name] = engine_cls


def list_engines() -> list[dict]:
    """List all registered engines with metadata."""
    result = []
    for name, cls in _engine_registry.items():
        inst = cls()
        result.append({
            "name": name,
            "display_name": inst.get_display_name(),
            "description": inst.get_description(),
            "stages": [{"id": s["id"], "name": s["name"]} for s in inst.get_stages()],
        })
    return result
