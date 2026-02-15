"""Point cloud metadata container."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Metadata:
    """Metadata associated with a point cloud.

    Attributes:
        source_file: Original file path this point cloud was read from.
        source_format: File format identifier (e.g. "las", "ply", "csv").
        creation_date: When this point cloud data was created/processed.
        software: Software that created/processed this data.
        point_format_id: LAS point format ID (0-10), if applicable.
        file_version: LAS file version (e.g. "1.4"), if applicable.
        extra: Free-form metadata dictionary.
    """

    source_file: str | None = None
    source_format: str | None = None
    creation_date: datetime | None = None
    software: str = "pywolken"
    point_format_id: int | None = None
    file_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> Metadata:
        """Return a shallow copy."""
        return Metadata(
            source_file=self.source_file,
            source_format=self.source_format,
            creation_date=self.creation_date,
            software=self.software,
            point_format_id=self.point_format_id,
            file_version=self.file_version,
            extra=dict(self.extra),
        )
