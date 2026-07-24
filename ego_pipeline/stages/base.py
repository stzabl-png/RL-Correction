"""PipelineStage - abstract base class for all pipeline stages."""
from __future__ import annotations
from abc import ABC, abstractmethod
from ego_pipeline.context import EgoContext


class PipelineStage(ABC):
    """Base class for every stage in the ego pipeline."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, ctx: EgoContext) -> EgoContext:
        ...

    def check_deps(self, ctx: EgoContext) -> list[str]:
        """Return field names that are required but missing."""
        return []
