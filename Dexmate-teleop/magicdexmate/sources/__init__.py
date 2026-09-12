from magicdexmate.sources.base import GloveSource, LatestSlot
from magicdexmate.sources.mock_source import MockGloveSource


def make_source(kind: str, hand: str, **kwargs) -> GloveSource:
    """Factory: kind in {"mock", "wuji"}. Extra kwargs go to the source ctor."""
    if kind == "mock":
        return MockGloveSource(hand=hand, **kwargs)
    if kind == "wuji":
        from magicdexmate.sources.wuji_source import WujiGloveSource

        return WujiGloveSource(hand=hand, **kwargs)
    raise ValueError(f"unknown source kind: {kind}")
