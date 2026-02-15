"""LAS/LAZ reader and writer using laspy[lazrs]."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import laspy
import laspy.header
import numpy as np

from pywolken.core.metadata import Metadata
from pywolken.core.pointcloud import PointCloud
from pywolken.io.base import Reader, Writer

# Mapping from laspy dimension names to pywolken standard names
_LASPY_DIM_MAP = {
    "x": "X",
    "y": "Y",
    "z": "Z",
    "intensity": "Intensity",
    "return_number": "ReturnNumber",
    "number_of_returns": "NumberOfReturns",
    "classification": "Classification",
    "scan_angle_rank": "ScanAngleRank",
    "scan_angle": "ScanAngleRank",
    "user_data": "UserData",
    "point_source_id": "PointSourceId",
    "gps_time": "GpsTime",
    "red": "Red",
    "green": "Green",
    "blue": "Blue",
    "nir": "NIR",
}


def _extract_crs(las: laspy.LasData) -> str | None:
    """Extract CRS from LAS VLRs as WKT string."""
    for vlr in las.vlrs:
        # OGC WKT CRS (GeoTIFF or WKT VLR)
        if vlr.record_id == 2112:
            try:
                return vlr.record_data.decode("utf-8").rstrip("\x00")
            except (UnicodeDecodeError, AttributeError):
                pass
    # Try header's CRS if available via laspy
    try:
        crs_vlrs = [v for v in las.vlrs if v.user_id == "LASF_Projection"]
        if crs_vlrs:
            for vlr in crs_vlrs:
                if vlr.record_id == 2112:
                    return vlr.record_data.decode("utf-8").rstrip("\x00")
    except Exception:
        pass
    return None


def _laspy_to_pointcloud(
    las: laspy.LasData,
    header: laspy.LasHeader | None = None,
) -> PointCloud:
    """Convert a laspy LasData or point record chunk to a PointCloud.

    Args:
        las: LasData object or ScaleAwarePointRecord chunk.
        header: LasHeader (required for chunks which lack their own header).
    """
    pc = PointCloud()

    for laspy_name, pw_name in _LASPY_DIM_MAP.items():
        try:
            data = las[laspy_name]
            # laspy returns ScaledArrayView for x/y/z — convert to plain ndarray
            arr = np.array(data, dtype=np.float64 if pw_name in ("X", "Y", "Z") else data.dtype)
            pc._arrays[pw_name] = arr
        except Exception:
            continue

    # Extra dimensions not in the standard mapping.
    # Note: point_format.dimension_names uses uppercase X,Y,Z (raw integers)
    # while lowercase x,y,z gives scaled floats. We must skip both cases.
    known_laspy = set(_LASPY_DIM_MAP.keys())
    known_pywolken = set(_LASPY_DIM_MAP.values())
    for dim_name in las.point_format.dimension_names:
        if dim_name not in known_laspy and dim_name not in known_pywolken:
            try:
                pc._arrays[dim_name] = np.array(las[dim_name])
            except Exception:
                continue

    # Metadata — use header if provided (for chunks), else from las object
    hdr = header if header is not None else getattr(las, "header", None)
    if hdr is not None:
        pc.metadata = Metadata(
            source_format="las",
            point_format_id=hdr.point_format.id,
            file_version=f"{hdr.version.major}.{hdr.version.minor}",
            software=hdr.generating_software or "pywolken",
        )
        # CRS from header's VLRs
        if hasattr(las, "vlrs"):
            pc.crs = _extract_crs(las)
    else:
        pc.metadata = Metadata(source_format="las")

    return pc


class LasReader(Reader):
    """LAS/LAZ reader using laspy with lazrs backend for LAZ decompression."""

    def read(self, path: str, **options: Any) -> PointCloud:
        """Read a LAS or LAZ file.

        Args:
            path: Path to .las or .laz file.

        Returns:
            PointCloud with all available dimensions.
        """
        las = laspy.read(path)
        pc = _laspy_to_pointcloud(las)
        pc.metadata.source_file = str(Path(path).resolve())
        return pc

    def read_chunked(
        self, path: str, chunk_size: int = 1_000_000, **options: Any
    ) -> Iterator[PointCloud]:
        """Stream-read a LAS/LAZ file in chunks.

        Uses laspy's built-in chunk iterator for memory-efficient reading.
        """
        with laspy.open(path) as las_reader:
            header = las_reader.header
            for chunk in las_reader.chunk_iterator(chunk_size):
                pc = _laspy_to_pointcloud(chunk, header=header)
                pc.metadata.source_file = str(Path(path).resolve())
                yield pc

    @classmethod
    def extensions(cls) -> list[str]:
        return [".las", ".laz"]

    @classmethod
    def type_name(cls) -> str:
        return "readers.las"


class LasWriter(Writer):
    """LAS/LAZ writer using laspy."""

    def write(self, pc: PointCloud, path: str, **options: Any) -> int:
        """Write a PointCloud to LAS or LAZ format.

        Args:
            pc: PointCloud to write.
            path: Output path. Use .laz extension for compressed output.
            **options:
                point_format_id: LAS point format (default: auto-detect).
                file_version: LAS version string like "1.4" (default: "1.4").

        Returns:
            Number of points written.
        """
        if pc.num_points == 0:
            raise ValueError("Cannot write empty PointCloud")

        point_format_id = options.get(
            "point_format_id",
            pc.metadata.point_format_id if pc.metadata.point_format_id is not None else self._detect_format(pc),
        )
        # Auto-select LAS version based on point format
        default_version = "1.4" if point_format_id >= 6 else "1.2"
        file_version = options.get("file_version", default_version)

        major, minor = (int(x) for x in file_version.split("."))
        header = laspy.LasHeader(
            point_format=point_format_id,
            version=laspy.header.Version(major=major, minor=minor),
        )
        header.generating_software = "pywolken"

        # Set CRS via WKT VLR
        if pc.crs:
            try:
                wkt_bytes = pc.crs.encode("utf-8")
                vlr = laspy.VLR(
                    user_id="LASF_Projection",
                    record_id=2112,
                    description="OGC WKT Coordinate System",
                    record_data=wkt_bytes,
                )
                header.vlrs.append(vlr)
            except Exception:
                pass

        # Set offsets and scales from data
        if "X" in pc and "Y" in pc and "Z" in pc:
            header.offsets = np.array([
                np.floor(np.min(pc["X"])),
                np.floor(np.min(pc["Y"])),
                np.floor(np.min(pc["Z"])),
            ])
            header.scales = np.array([0.001, 0.001, 0.001])

        las = laspy.LasData(header)

        # Reverse mapping: pywolken name -> laspy name (prefer first match)
        pw_to_laspy: dict[str, str] = {}
        for lname, pname in _LASPY_DIM_MAP.items():
            if pname not in pw_to_laspy:
                pw_to_laspy[pname] = lname

        for dim_name in pc.dimensions:
            laspy_name = pw_to_laspy.get(dim_name, dim_name)
            try:
                las[laspy_name] = pc[dim_name]
            except Exception:
                # Try alternate name for scan angle
                if dim_name == "ScanAngleRank":
                    try:
                        las["scan_angle"] = pc[dim_name]
                    except Exception:
                        pass
                continue

        las.write(path)
        return pc.num_points

    def _detect_format(self, pc: PointCloud) -> int:
        """Auto-detect appropriate LAS point format based on available dimensions."""
        has_color = "Red" in pc and "Green" in pc and "Blue" in pc
        has_gps = "GpsTime" in pc
        has_nir = "NIR" in pc

        if has_nir:
            return 8  # Point format 8: GPS + RGB + NIR
        if has_color and has_gps:
            return 7  # Point format 7: GPS + RGB (1.4)
        if has_color:
            return 2  # Point format 2: RGB
        if has_gps:
            return 6  # Point format 6: GPS time (1.4)
        return 0  # Point format 0: basic

    @classmethod
    def extensions(cls) -> list[str]:
        return [".las", ".laz"]

    @classmethod
    def type_name(cls) -> str:
        return "writers.las"
