# pywolken Documentation

> **Version:** 0.1.0
> **License:** MIT
> **Python:** >= 3.10
> **Repository:** [github.com/54/pywolken](https://github.com/54/pywolken)

pywolken is a pure-Python point cloud processing library — a complete alternative to PDAL.
It reads LAS/LAZ/PLY/CSV files, applies filter chains via JSON pipelines, generates DEMs and hillshades, creates 3D meshes, and scales to billions of points via chunked streaming and optional Dask parallelism. No C++ compilation required.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [PointCloud](#pointcloud)
  - [Bounds](#bounds)
  - [Metadata](#metadata)
  - [Dimensions](#dimensions)
- [I/O — Reading and Writing](#io--reading-and-writing)
  - [Convenience Functions](#convenience-functions)
  - [LAS/LAZ](#laslaz)
  - [PLY](#ply)
  - [CSV/TXT/XYZ](#csvtxtxyz)
  - [GeoTIFF (DEM Writer)](#geotiff-dem-writer)
  - [I/O Registry](#io-registry)
- [Filters](#filters)
  - [Filter Registry](#filter-registry)
  - [filters.range](#filtersrange)
  - [filters.crop](#filterscrop)
  - [filters.merge](#filtersmerge)
  - [filters.decimation](#filtersdecimation)
  - [filters.assign](#filtersassign)
  - [filters.expression](#filtersexpression)
  - [filters.reprojection](#filtersreprojection)
  - [filters.ground](#filtersground)
  - [filters.hag](#filtershag)
  - [filters.outlier](#filtersoutlier)
  - [filters.normal](#filtersnormal)
  - [filters.voxel](#filtersvoxel)
  - [filters.cluster](#filterscluster)
  - [filters.colorize](#filterscolorize)
  - [filters.sort](#filterssort)
- [Pipeline Engine](#pipeline-engine)
  - [JSON Pipeline Format](#json-pipeline-format)
  - [Pipeline Class](#pipeline-class)
  - [StreamingPipeline](#streamingpipeline)
- [Raster Module](#raster-module)
  - [DEM Generation](#dem-generation)
  - [Hillshade](#hillshade)
  - [GeoTIFF Export](#geotiff-export)
- [Mesh Module](#mesh-module)
  - [Triangulation](#triangulation)
  - [Mesh Export](#mesh-export)
- [Parallel Processing (Dask)](#parallel-processing-dask)
- [CLI Reference](#cli-reference)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Examples](#examples)

---

## Installation

```bash
# Core (LAS/LAZ, filters, pipelines)
pip install pywolken

# With raster support (DEM, hillshade, GeoTIFF export)
pip install pywolken[raster]

# With Dask for parallel processing
pip install pywolken[dask]

# Everything
pip install pywolken[all]

# Development
pip install pywolken[dev]
```

### Core dependencies (installed automatically)

| Package | Purpose |
|---------|---------|
| `numpy >= 1.24` | Array operations, core data representation |
| `laspy[lazrs] >= 2.5` | LAS/LAZ I/O (lazrs = Rust-based LAZ codec, ships as wheel) |
| `scipy >= 1.10` | KDTree spatial indexing, Delaunay triangulation, interpolation |
| `pyproj >= 3.5` | CRS definitions and coordinate reprojection |

### Optional dependencies

| Extra | Package | Purpose |
|-------|---------|---------|
| `raster` | `rasterio >= 1.3` | GeoTIFF export, raster colorization |
| `dask` | `dask[array] >= 2023.0` | Parallel/distributed chunk processing |
| `viz` | `matplotlib >= 3.7, plotly >= 5.0` | Visualization |
| `mesh` | `open3d >= 0.17` | Advanced mesh processing (Poisson reconstruction) |

---

## Quick Start

```python
import pywolken

# Read any supported format (auto-detected from extension)
pc = pywolken.read("terrain.laz")
print(pc)
# PointCloud(45,266,951 points, dims=[X, Y, Z, Intensity, Classification, GpsTime])

# Access dimensions as NumPy arrays
print(pc["X"].mean())     # 399521.43
print(pc["Z"].min())      # 318.02
print(pc.bounds)          # Bounds(x=[399000.00, 400000.00], ...)
print(pc.crs)             # EPSG:25832

# Filter with boolean mask
ground = pc.mask(pc["Classification"] == 2)
print(ground.num_points)  # 23,354,571

# Write to any format
pywolken.write(ground, "ground.laz")
pywolken.write(ground, "ground.ply", format="ascii")
pywolken.write(ground, "ground.csv", delimiter=";")

# JSON pipeline (PDAL-compatible)
import json
pipeline = pywolken.Pipeline(json.dumps({
    "pipeline": [
        "input.laz",
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {"type": "filters.decimation", "step": 10},
        "output.las"
    ]
}))
count = pipeline.execute()
print(f"Processed {count:,} points")
```

---

## Core Concepts

### PointCloud

The central data structure. Stores point dimensions as a dict of NumPy arrays (columnar layout). Each dimension (X, Y, Z, Intensity, Classification, etc.) is a separate contiguous array.

**Module:** `pywolken.core.pointcloud`

```python
from pywolken import PointCloud
import numpy as np

# Create empty and add dimensions
pc = PointCloud()
pc["X"] = np.array([1.0, 2.0, 3.0])
pc["Y"] = np.array([4.0, 5.0, 6.0])
pc["Z"] = np.array([7.0, 8.0, 9.0])

# Create from dict (preferred)
pc = PointCloud.from_dict({
    "X": np.array([1.0, 2.0, 3.0]),
    "Y": np.array([4.0, 5.0, 6.0]),
    "Z": np.array([7.0, 8.0, 9.0]),
    "Classification": np.array([2, 2, 6], dtype=np.uint8),
}, crs="EPSG:25832")

# Create from NumPy structured array
arr = np.zeros(100, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
pc = PointCloud.from_numpy(arr)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_points` | `int` | Number of points |
| `dimensions` | `list[str]` | List of dimension names |
| `bounds` | `Bounds` | 3D bounding box (requires X, Y, Z) |
| `metadata` | `Metadata` | Source file info, format, etc. |
| `crs` | `str \| None` | Coordinate reference system (EPSG or WKT) |

#### Methods

```python
# Array access
pc["X"]                          # Get dimension as np.ndarray
pc["X"] = new_array              # Set dimension
"X" in pc                        # Check if dimension exists
len(pc)                          # Number of points

# Filtering
filtered = pc.mask(pc["Z"] > 100)          # Boolean mask → new PointCloud
chunk = pc.slice(0, 1000)                  # Slice by index range

# Iteration
for chunk in pc.iter_chunks(1_000_000):    # Iterate in chunks
    process(chunk)

# Dimension management
pc.add_dimension("NewDim", dtype=np.float32, fill=0.0)
pc.remove_dimension("NewDim")

# Conversion
structured_arr = pc.to_numpy()             # → NumPy structured array
dim_dict = pc.to_dict()                    # → dict[str, np.ndarray]

# Operations
copy = pc.copy()                           # Deep copy
merged = pc.merge(other_pc)                # Merge (common dimensions only)
```

---

### Bounds

Frozen dataclass representing a 3D axis-aligned bounding box.

**Module:** `pywolken.core.bounds`

```python
from pywolken import Bounds

b = Bounds(minx=0, miny=0, minz=0, maxx=100, maxy=100, maxz=50)
b = Bounds.from_arrays(x_arr, y_arr, z_arr)
b = Bounds.from_pointcloud(pc)

b.contains_point(50, 50, 25)     # True
b.contains_2d(50, 50)            # True
b.intersects(other_bounds)       # True/False
union = b.union(other_bounds)    # Enclosing box

b.width     # maxx - minx
b.height    # maxy - miny
b.depth     # maxz - minz
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `minx` | `float` | Minimum X |
| `miny` | `float` | Minimum Y |
| `minz` | `float` | Minimum Z |
| `maxx` | `float` | Maximum X |
| `maxy` | `float` | Maximum Y |
| `maxz` | `float` | Maximum Z |

---

### Metadata

Dataclass with point cloud source information.

**Module:** `pywolken.core.metadata`

```python
from pywolken import Metadata

m = Metadata(
    source_file="/data/terrain.laz",
    source_format="las",
    point_format_id=6,
    file_version="1.4",
    software="pywolken",
)
copy = m.copy()
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_file` | `str \| None` | `None` | Original file path |
| `source_format` | `str \| None` | `None` | Format: "las", "ply", "csv" |
| `creation_date` | `datetime \| None` | `None` | File creation timestamp |
| `software` | `str` | `"pywolken"` | Generating software |
| `point_format_id` | `int \| None` | `None` | LAS point format (0-10) |
| `file_version` | `str \| None` | `None` | LAS version ("1.2", "1.4") |
| `extra` | `dict[str, Any]` | `{}` | Arbitrary extra metadata |

---

### Dimensions

Standard LAS dimension definitions and ASPRS classification codes.

**Module:** `pywolken.core.dimensions`

```python
from pywolken.core.dimensions import STANDARD_DIMENSIONS, CLASSIFICATION_CODES, get_dtype

# Standard dimension → dtype mapping
STANDARD_DIMENSIONS = {
    "X": float64, "Y": float64, "Z": float64,
    "Intensity": uint16,
    "ReturnNumber": uint8, "NumberOfReturns": uint8,
    "Classification": uint8,
    "ScanAngleRank": float32,
    "UserData": uint8, "PointSourceId": uint16,
    "GpsTime": float64,
    "Red": uint16, "Green": uint16, "Blue": uint16, "NIR": uint16,
}

# ASPRS classification codes
CLASSIFICATION_CODES = {
    0: "Created, Never Classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (Noise)",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    17: "Bridge Deck",
    18: "High Noise",
}

dtype = get_dtype("Classification")  # uint8
dtype = get_dtype("CustomDim")       # float64 (default)
```

---

## I/O — Reading and Writing

### Convenience Functions

**Module:** `pywolken.io.registry`
**Also exported from:** `pywolken` (top-level)

```python
import pywolken

# Auto-detect format from file extension
pc = pywolken.read("file.laz")
pc = pywolken.read("file.ply")
pc = pywolken.read("file.csv", delimiter=";")

# Write (format from extension)
pywolken.write(pc, "output.las")
pywolken.write(pc, "output.ply", format="ascii")
pywolken.write(pc, "output.csv", precision=3)

# Chunked reading for large files
for chunk in pywolken.read_chunked("huge.laz", chunk_size=2_000_000):
    process(chunk)
```

#### `read(path, **options) -> PointCloud`

Read a point cloud file. Format auto-detected from extension.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | File path (.las, .laz, .ply, .csv, .txt, .xyz) |
| `**options` | | Format-specific options (see per-format docs below) |

#### `write(pc, path, **options) -> int`

Write a point cloud. Returns number of points written.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pc` | `PointCloud` | Point cloud to write |
| `path` | `str` | Output path (format from extension) |
| `**options` | | Format-specific options |

#### `read_chunked(path, chunk_size=1_000_000, **options) -> Iterator[PointCloud]`

Stream-read a file in chunks. LAS/LAZ uses native chunked reading for true streaming.

---

### LAS/LAZ

**Reader type:** `readers.las`
**Writer type:** `writers.las`
**Extensions:** `.las`, `.laz`

Uses `laspy[lazrs]` — lazrs provides Rust-based LAZ compression shipped as a pre-built wheel.

```python
from pywolken.io.las import LasReader, LasWriter

# Read
reader = LasReader()
pc = reader.read("terrain.laz")

# Chunked read (memory-efficient)
for chunk in reader.read_chunked("huge.laz", chunk_size=1_000_000):
    process(chunk)

# Write
writer = LasWriter()
writer.write(pc, "output.las")
writer.write(pc, "output.laz")  # compressed
writer.write(pc, "output.las", point_format_id=6, file_version="1.4")
```

#### Writer Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `point_format_id` | `int` | auto-detect | LAS point format (0-10) |
| `file_version` | `str` | auto | "1.2" for formats 0-5, "1.4" for formats 6-10 |

**Auto-detection logic:**
- Format 8 → has NIR
- Format 7 → has RGB + GpsTime
- Format 2 → has RGB
- Format 6 → has GpsTime
- Format 0 → basic

**Dimension mapping (laspy lowercase → pywolken uppercase):**

| laspy | pywolken |
|-------|----------|
| `x`, `y`, `z` | `X`, `Y`, `Z` |
| `intensity` | `Intensity` |
| `return_number` | `ReturnNumber` |
| `number_of_returns` | `NumberOfReturns` |
| `classification` | `Classification` |
| `scan_angle_rank` / `scan_angle` | `ScanAngleRank` |
| `user_data` | `UserData` |
| `point_source_id` | `PointSourceId` |
| `gps_time` | `GpsTime` |
| `red`, `green`, `blue` | `Red`, `Green`, `Blue` |
| `nir` | `NIR` |

Extra dimensions not in this mapping are preserved with their original names.

---

### PLY

**Reader type:** `readers.ply`
**Writer type:** `writers.ply`
**Extensions:** `.ply`

Stanford PLY format. Supports ASCII, binary little-endian, and binary big-endian.

```python
from pywolken.io.ply import PlyReader, PlyWriter

pc = PlyReader().read("model.ply")
PlyWriter().write(pc, "output.ply", format="ascii")
PlyWriter().write(pc, "output.ply", format="binary_little_endian")
```

#### Writer Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `format` | `str` | `"binary_little_endian"` | `"ascii"`, `"binary_little_endian"`, or `"binary_big_endian"` |

**PLY ↔ pywolken dimension mapping:**

| PLY | pywolken |
|-----|----------|
| `x`, `y`, `z` | `X`, `Y`, `Z` |
| `nx`, `ny`, `nz` | `NormalX`, `NormalY`, `NormalZ` |
| `red`, `green`, `blue`, `alpha` | `Red`, `Green`, `Blue`, `Alpha` |
| `intensity` | `Intensity` |
| `classification` | `Classification` |
| `gps_time` | `GpsTime` |

**Supported PLY data types:** `char`, `uchar`, `short`, `ushort`, `int`, `uint`, `float`, `double`, `int8`-`int64`, `uint8`-`uint64`, `float32`, `float64`

---

### CSV/TXT/XYZ

**Reader type:** `readers.csv`
**Writer type:** `writers.csv`
**Extensions:** `.csv`, `.txt`, `.xyz`

Delimited text format. Auto-detects delimiter from `,`, `;`, `\t`, ` `.

```python
from pywolken.io.csv import CsvReader, CsvWriter

# Auto-detect delimiter and header
pc = CsvReader().read("points.csv")

# Explicit options
pc = CsvReader().read("points.txt", delimiter="\t", skip=2)

# No header row — provide column names
pc = CsvReader().read("points.xyz", header="X,Y,Z,Intensity")

# Write
CsvWriter().write(pc, "output.csv")
CsvWriter().write(pc, "output.csv", delimiter=";", precision=3, header=True)
```

#### Reader Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `delimiter` | `str` | auto-detect | Field separator |
| `header` | `str` | `None` | Comma-separated column names (if file has no header) |
| `skip` | `int` | `0` | Lines to skip before data |

#### Writer Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `delimiter` | `str` | `","` | Field separator |
| `header` | `bool` | `True` | Write header row |
| `precision` | `int` | `6` | Decimal places for float values |

---

### GeoTIFF (DEM Writer)

**Writer type:** `writers.gdal`
**Extensions:** `.tif`, `.tiff`

Generates a DEM from a point cloud and writes it as a GeoTIFF. Requires `rasterio`.

```python
from pywolken.io.gdal import GdalWriter

writer = GdalWriter()
writer.write(pc, "dem.tif", resolution=0.5, output_type="idw")
```

#### Writer Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `resolution` | `float` | **required** | Grid cell size in CRS units |
| `output_type` | `str` | `"idw"` | Interpolation: `"idw"`, `"mean"`, `"nearest"`, `"tin"` |
| `window_size` | `int` | `6` | IDW search window (cells) |
| `power` | `float` | `2.0` | IDW distance power |
| `nodata` | `float` | `-9999.0` | Nodata fill value |
| `gdalopts` | `str` | `""` | GDAL creation options: `"COMPRESS=LZW,TILED=YES"` |

---

### I/O Registry

Format auto-detection and plugin registration.

```python
from pywolken.io.registry import io_registry

# Register a custom reader/writer
io_registry.register_reader(MyCustomReader)
io_registry.register_writer(MyCustomWriter)

# Get reader/writer by file extension
reader = io_registry.get_reader("file.laz")     # → LasReader
writer = io_registry.get_writer("output.ply")    # → PlyWriter

# Get by pipeline type name
reader = io_registry.get_reader_by_type("readers.las")
writer = io_registry.get_writer_by_type("writers.csv")
```

---

## Filters

All filters follow the same pattern: instantiate with options, call `filter(pc)` to get a new PointCloud. Filters are immutable — they never modify the input.

```python
from pywolken.filters.range import RangeFilter

f = RangeFilter(limits="Classification[2:2]")
result = f.filter(pc)
```

### Filter Registry

All 15 built-in filters are registered and can be created by type name:

```python
from pywolken.filters.registry import get_filter, filter_registry

# Create by type name
f = get_filter("filters.range", limits="Classification[2:2]")
result = f.filter(pc)

# List all available filters
print(filter_registry.available)
# ['filters.range', 'filters.crop', 'filters.merge', 'filters.decimation',
#  'filters.assign', 'filters.expression', 'filters.reprojection',
#  'filters.ground', 'filters.hag', 'filters.outlier', 'filters.normal',
#  'filters.voxel', 'filters.cluster', 'filters.colorize', 'filters.sort']
```

---

### filters.range

Filter points by dimension value ranges. PDAL-compatible syntax.

**Type name:** `filters.range`

```python
from pywolken.filters.range import RangeFilter

f = RangeFilter(limits="Classification[2:2]")           # Exact class 2
f = RangeFilter(limits="Z[100:500]")                     # Z between 100-500
f = RangeFilter(limits="Z[100:]")                        # Z >= 100
f = RangeFilter(limits="Z[:500]")                        # Z <= 500
f = RangeFilter(limits="Classification![7:7]")           # Exclude noise (class 7)
f = RangeFilter(limits="Classification[2:2],Z[100:]")    # Multiple (AND)
result = f.filter(pc)
```

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `limits` | `str` | yes | PDAL-style range expression |

**Syntax:** `DimensionName[min:max]` — brackets inclusive. Prefix with `!` to negate. Multiple conditions comma-separated (AND logic).

---

### filters.crop

Spatial bounding box crop.

**Type name:** `filters.crop`

```python
from pywolken.filters.crop import CropFilter

# PDAL-style bounds string
f = CropFilter(bounds="([399000, 400000], [5600000, 5601000])")

# 3D bounds
f = CropFilter(bounds="([399000, 400000], [5600000, 5601000], [100, 500])")

# Explicit values
f = CropFilter(minx=399000, maxx=400000, miny=5600000, maxy=5601000)

result = f.filter(pc)
```

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `bounds` | `str` | one of | PDAL syntax: `"([xmin,xmax],[ymin,ymax])"` |
| `minx`, `maxx`, `miny`, `maxy` | `float` | one of | Explicit 2D bounds |
| `minz`, `maxz` | `float` | no | Optional Z bounds |

---

### filters.merge

Merge multiple point clouds. Common dimensions only.

**Type name:** `filters.merge`

```python
from pywolken.filters.merge import MergeFilter

f = MergeFilter()
f.add_input(other_pc1)
f.add_input(other_pc2)
result = f.filter(pc)  # Merges pc + other_pc1 + other_pc2
```

---

### filters.decimation

Reduce point count by subsampling.

**Type name:** `filters.decimation`

```python
from pywolken.filters.decimation import DecimationFilter

f = DecimationFilter(step=10)                       # Every 10th point
f = DecimationFilter(fraction=0.1)                  # Random 10%
f = DecimationFilter(count=100000)                  # Exactly 100k points
f = DecimationFilter(fraction=0.5, seed=42)         # Reproducible random
result = f.filter(pc)
```

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `step` | `int` | one of | Keep every Nth point |
| `fraction` | `float` | one of | Keep this fraction (0.0, 1.0] |
| `count` | `int` | one of | Keep exactly this many |
| `seed` | `int` | no | Random seed for reproducibility |

---

### filters.assign

Set dimension values to constants.

**Type name:** `filters.assign`

```python
from pywolken.filters.assign import AssignFilter

f = AssignFilter(assignment="Classification=2,Intensity=0")
f = AssignFilter(value={"Classification": 2, "Intensity": 0})
result = f.filter(pc)
```

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `assignment` | `str` | one of | `"Dim=value,Dim=value"` |
| `value` | `dict` | one of | `{"Dim": value}` |

Creates new dimensions if they don't exist.

---

### filters.expression

Filter points by boolean expressions.

**Type name:** `filters.expression`

```python
from pywolken.filters.expression import ExpressionFilter

f = ExpressionFilter(expression="Classification == 2")
f = ExpressionFilter(expression="Z > 100 AND Z < 500")
f = ExpressionFilter(expression="Classification == 2 OR Classification == 6")
f = ExpressionFilter(where="Intensity >= 500")  # 'where' alias
result = f.filter(pc)
```

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `expression` | `str` | one of | Boolean expression |
| `where` | `str` | one of | Alias for expression |

**Operators:** `==`, `!=`, `>`, `<`, `>=`, `<=`
**Combinators:** `AND`, `OR`

---

### filters.reprojection

Transform coordinates between CRS. Requires `pyproj`.

**Type name:** `filters.reprojection`

```python
from pywolken.filters.reprojection import ReprojectionFilter

f = ReprojectionFilter(out_srs="EPSG:4326")                       # Auto source CRS
f = ReprojectionFilter(in_srs="EPSG:25832", out_srs="EPSG:4326")  # Explicit source
result = f.filter(pc)
```

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `out_srs` | `str` | yes | Target CRS (e.g., `"EPSG:4326"`) |
| `in_srs` | `str` | no | Source CRS (uses `pc.crs` if omitted) |

Updates `pc.crs` on the result.

---

### filters.ground

SMRF (Simple Morphological Filter) ground classification. Sets `Classification=2` for ground points, `Classification=1` for non-ground.

**Type name:** `filters.ground`

```python
from pywolken.filters.ground import GroundFilter

f = GroundFilter()
f = GroundFilter(cell_size=1.0, slope=0.15, window_max=18.0, threshold=0.5)
result = f.filter(pc)
ground_only = result.mask(result["Classification"] == 2)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cell_size` | `float` | `1.0` | Initial minimum surface grid cell size |
| `slope` | `float` | `0.15` | Maximum terrain slope |
| `window_max` | `float` | `18.0` | Maximum morphological window size |
| `threshold` | `float` | `0.5` | Height difference threshold |
| `scalar` | `float` | `1.25` | Slope scaling factor |

---

### filters.hag

Height Above Ground — computes each point's elevation relative to the ground surface.

**Type name:** `filters.hag`

**Prerequisite:** Point cloud must have `Classification` dimension with ground points (`Classification == 2`).

```python
from pywolken.filters.hag import HagFilter

f = HagFilter(neighbors=3)
result = f.filter(pc)
print(result["HeightAboveGround"])  # New dimension
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `neighbors` | `int` | `3` | Ground neighbors to average for interpolation |

**Output:** Adds `HeightAboveGround` dimension (float64).

---

### filters.outlier

Statistical or radius-based outlier removal using KDTree.

**Type name:** `filters.outlier`

```python
from pywolken.filters.outlier import OutlierFilter

# Statistical: remove if mean_dist > mean + multiplier * stddev
f = OutlierFilter(method="statistical", mean_k=8, multiplier=2.0)

# Radius: remove if fewer than min_k neighbors within radius
f = OutlierFilter(method="radius", radius=5.0, min_k=3)

result = f.filter(pc)  # Returns PointCloud without outliers
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `method` | `str` | `"statistical"` | `"statistical"` or `"radius"` |
| `mean_k` | `int` | `8` | (statistical) Neighbors to analyze |
| `multiplier` | `float` | `2.0` | (statistical) Std deviation multiplier |
| `radius` | `float` | — | (radius) Search radius |
| `min_k` | `int` | `2` | (radius) Min neighbors required |

---

### filters.normal

Estimate surface normals via PCA on KDTree neighborhoods.

**Type name:** `filters.normal`

```python
from pywolken.filters.normal import NormalFilter

f = NormalFilter(k=8)
result = f.filter(pc)
print(result["NormalX"], result["NormalY"], result["NormalZ"])
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `k` | `int` | `8` | Neighbors for PCA normal estimation |

**Output:** Adds `NormalX`, `NormalY`, `NormalZ` dimensions (float64). Normals are oriented upward (NormalZ > 0).

---

### filters.voxel

Voxel grid downsampling. Replaces all points within each voxel cell with a single centroid.

**Type name:** `filters.voxel`

```python
from pywolken.filters.voxel import VoxelFilter

f = VoxelFilter(cell_size=0.5)
result = f.filter(pc)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cell_size` | `float` | `1.0` | Voxel edge length in CRS units |
| `cell` | `float` | — | Alias for `cell_size` |

X, Y, Z are averaged (centroid). Other dimensions take the first point's value.

---

### filters.cluster

DBSCAN-based point cloud clustering.

**Type name:** `filters.cluster`

```python
from pywolken.filters.cluster import ClusterFilter

f = ClusterFilter(tolerance=5.0, min_points=10)
result = f.filter(pc)
cluster_ids = result["ClusterID"]  # 0 = noise, 1+ = cluster ID
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tolerance` | `float` | `1.0` | Maximum distance between cluster neighbors |
| `min_points` | `int` | `10` | Minimum points per cluster |
| `is3d` | `bool` | `True` | Use 3D distance (False = 2D XY only) |

**Output:** Adds `ClusterID` dimension (int64). `0` = noise/unassigned.

---

### filters.colorize

Assign RGB values from a raster overlay (e.g., orthophoto). Requires `rasterio`.

**Type name:** `filters.colorize`

```python
from pywolken.filters.colorize import ColorizeFilter

f = ColorizeFilter(raster="/data/orthophoto.tif")
result = f.filter(pc)
```

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `raster` | `str` | yes | Path to raster file (GeoTIFF, etc.) |

**Output:** Adds/updates `Red`, `Green`, `Blue` dimensions (uint16). 8-bit rasters are scaled to 16-bit (×257).

---

### filters.sort

Spatial sorting for improved cache locality.

**Type name:** `filters.sort`

```python
from pywolken.filters.sort import SortFilter

f = SortFilter(order="morton")   # Z-order curve (default)
f = SortFilter(order="xyz")      # Lexicographic X, then Y, then Z
result = f.filter(pc)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `order` | `str` | `"morton"` | `"morton"` (Z-order curve) or `"xyz"` (lexicographic) |

Morton sort maps 2D XY coordinates to a 1D Z-order curve using 16-bit precision, giving spatially nearby points nearby array positions.

---

## Pipeline Engine

### JSON Pipeline Format

pywolken uses PDAL-compatible JSON pipelines. A pipeline is a list of stages: readers, filters, and writers.

```json
{
  "pipeline": [
    "input.laz",
    {"type": "filters.range", "limits": "Classification[2:2]"},
    {"type": "filters.decimation", "step": 10},
    {"type": "filters.voxel", "cell_size": 0.5},
    "output.las"
  ]
}
```

#### Stage types

**Bare strings** are auto-detected as readers (first) or writers (subsequent):
```json
["input.laz", "output.las"]
```

**Dict stages** with explicit `type`:
```json
{"type": "readers.las", "filename": "input.laz"}
{"type": "filters.range", "limits": "Z[100:]"}
{"type": "writers.las", "filename": "output.las"}
```

#### Supported reader types
`readers.las`, `readers.ply`, `readers.csv`

#### Supported writer types
`writers.las`, `writers.ply`, `writers.csv`, `writers.gdal`

#### Supported filter types
`filters.range`, `filters.crop`, `filters.merge`, `filters.decimation`, `filters.assign`, `filters.expression`, `filters.reprojection`, `filters.ground`, `filters.hag`, `filters.outlier`, `filters.normal`, `filters.voxel`, `filters.cluster`, `filters.colorize`, `filters.sort`

---

### Pipeline Class

**Module:** `pywolken.pipeline.pipeline`

```python
from pywolken import Pipeline
import json

# From JSON string
p = Pipeline(json.dumps({
    "pipeline": [
        "input.laz",
        {"type": "filters.ground"},
        {"type": "filters.range", "limits": "Classification[2:2]"},
        "ground.laz"
    ]
}))

# Validate
errors = p.validate()  # [] if valid

# Execute
count = p.execute()     # Returns total points processed

# Access results
pc = p.result           # PointCloud from last execution
arrays = p.arrays       # dict[str, np.ndarray]
meta = p.metadata       # dict with source info

# Serialize back to JSON
json_str = p.to_json()
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `validate()` | `list[str]` | Error messages (empty = valid) |
| `execute()` | `int` | Run pipeline, return point count |
| `to_json()` | `str` | Serialize to JSON string |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `result` | `PointCloud \| None` | Result from last execution |
| `arrays` | `dict[str, np.ndarray] \| None` | Point data from last execution |
| `metadata` | `dict[str, Any]` | Metadata from last execution |

---

### StreamingPipeline

Memory-bounded pipeline that processes data in chunks. Reads chunks, applies per-chunk filters, writes incrementally. Suitable for files larger than available RAM.

**Module:** `pywolken.pipeline.streaming`

```python
from pywolken import StreamingPipeline
from pywolken.io.las import LasReader, LasWriter
from pywolken.filters.range import RangeFilter
from pywolken.filters.expression import ExpressionFilter

sp = StreamingPipeline(
    reader=LasReader(),
    input_path="huge.laz",
    writer=LasWriter(),
    output_path="filtered.laz",
    filters=[
        RangeFilter(limits="Classification[2:2]"),
        ExpressionFilter(expression="Z > 100"),
    ],
    chunk_size=2_000_000,
)

total = sp.execute()
print(f"Processed {sp.total_read:,} → {sp.total_written:,} points")

# Or iterate without writing
for chunk in sp.iter_results():
    process(chunk)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reader` | `Reader` | **required** | Reader instance |
| `input_path` | `str` | **required** | Input file path |
| `writer` | `Writer \| None` | `None` | Writer instance |
| `output_path` | `str \| None` | `None` | Output file path |
| `filters` | `list[Filter] \| None` | `None` | Filter chain |
| `chunk_size` | `int` | `1_000_000` | Points per chunk |
| `reader_options` | `dict` | `{}` | Extra reader options |
| `writer_options` | `dict` | `{}` | Extra writer options |

#### Streaming-compatible filters

These filters work correctly in per-chunk mode:
- `filters.range`, `filters.crop`, `filters.expression`, `filters.assign`
- `filters.decimation`, `filters.voxel`, `filters.sort`

These filters need global context and should **not** be used in streaming mode:
- `filters.ground`, `filters.cluster`, `filters.normal`, `filters.outlier`, `filters.hag`

---

## Raster Module

### DEM Generation

Generate digital elevation models from point clouds.

**Module:** `pywolken.raster.dem`

```python
from pywolken.raster.dem import create_dem

raster, transform = create_dem(
    pc,
    resolution=0.5,       # 0.5m grid cells
    method="idw",          # Interpolation method
    power=2.0,             # IDW distance power
)

print(raster.shape)        # (2000, 2000)
print(raster.dtype)        # float32
print(transform)           # {'xmin': ..., 'ymax': ..., 'resolution': 0.5, ...}
```

#### `create_dem(pc, resolution, method="idw", bounds=None, nodata=-9999.0, power=2.0, radius=None, window_size=6)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pc` | `PointCloud` | **required** | Point cloud with X, Y, Z |
| `resolution` | `float` | **required** | Grid cell size in CRS units |
| `method` | `str` | `"idw"` | `"idw"`, `"mean"`, `"nearest"`, or `"tin"` |
| `bounds` | `tuple` | `None` | `(xmin, ymin, xmax, ymax)` or auto from data |
| `nodata` | `float` | `-9999.0` | Nodata fill value |
| `power` | `float` | `2.0` | IDW distance power |
| `radius` | `float` | `None` | Search radius (auto if None) |
| `window_size` | `int` | `6` | IDW search window in grid cells |

**Returns:** `(raster, transform_info)` where:
- `raster`: 2D `float32` array (rows x cols, top-left origin)
- `transform_info`: dict with keys `xmin`, `ymax`, `resolution`, `nrows`, `ncols`, `crs`

#### Interpolation Methods

| Method | Description | Speed | Quality |
|--------|-------------|-------|---------|
| `"idw"` | Inverse Distance Weighting (batch KDTree) | Medium | Good |
| `"mean"` | Grid-cell mean (np.add.at binning) | Fast | Lower |
| `"nearest"` | Nearest-neighbor (KDTree) | Fast | Blocky |
| `"tin"` | Delaunay TIN + linear interpolation | Slow | Best |

---

### Hillshade

Compute shaded relief from a DEM using Horn's method.

**Module:** `pywolken.raster.hillshade`

```python
from pywolken.raster.hillshade import hillshade, multi_directional_hillshade

# Single-direction hillshade
hs = hillshade(
    dem=raster,
    resolution=0.5,
    azimuth=315.0,     # Light from NW
    altitude=45.0,     # 45 degrees above horizon
    z_factor=2.0,      # Vertical exaggeration
)
# hs.shape == dem.shape, hs.dtype == uint8

# Multi-directional (blended from 8 directions)
hs_multi = multi_directional_hillshade(
    dem=raster,
    resolution=0.5,
    z_factor=2.0,
)
```

#### `hillshade(dem, resolution, azimuth=315.0, altitude=45.0, z_factor=1.0, nodata=-9999.0)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dem` | `np.ndarray` | **required** | 2D float array |
| `resolution` | `float` | **required** | Cell size in CRS units |
| `azimuth` | `float` | `315.0` | Light direction (0=N, 90=E, 180=S, 270=W) |
| `altitude` | `float` | `45.0` | Light elevation (0=horizon, 90=zenith) |
| `z_factor` | `float` | `1.0` | Vertical exaggeration |
| `nodata` | `float` | `-9999.0` | Nodata value in DEM |

**Returns:** 2D `uint8` array (0-255 shaded relief).

#### `multi_directional_hillshade(dem, resolution, altitude=45.0, z_factor=1.0, nodata=-9999.0, directions=None, weights=None)`

Blends hillshades from multiple azimuth directions (default: 8 cardinal/ordinal).

---

### GeoTIFF Export

Write raster arrays to GeoTIFF. Requires `rasterio`.

**Module:** `pywolken.raster.export`

```python
from pywolken.raster.export import write_geotiff

write_geotiff(
    array=hillshade_array,
    path="hillshade.tif",
    transform_info=transform,     # from create_dem()
    nodata=None,                  # No nodata for hillshade
    compress="lzw",
    dtype="uint8",
)
```

#### `write_geotiff(array, path, transform_info, nodata=-9999.0, compress="lzw", dtype=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | `np.ndarray` | **required** | 2D raster array |
| `path` | `str` | **required** | Output file path |
| `transform_info` | `dict` | **required** | From `create_dem()` — needs `xmin`, `ymax`, `resolution`, `nrows`, `ncols`, `crs` |
| `nodata` | `float \| None` | `-9999.0` | Nodata value (None to omit) |
| `compress` | `str` | `"lzw"` | `"lzw"`, `"deflate"`, `"none"` |
| `dtype` | `str \| None` | `None` | Output dtype (auto from array) |

---

## Mesh Module

### Triangulation

2.5D Delaunay triangulation — project to XY, triangulate, use Z for 3D vertices.

**Module:** `pywolken.mesh.triangulate`

```python
from pywolken.mesh import triangulate_2d, Mesh

# Create mesh from point cloud
mesh = triangulate_2d(pc)
mesh = triangulate_2d(pc, max_edge_length=10.0)  # Remove long-edge triangles

print(mesh)  # Mesh(45,000 vertices, 89,000 faces)
```

#### `triangulate_2d(pc, max_edge_length=None) -> Mesh`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pc` | `PointCloud` | **required** | Point cloud with X, Y, Z |
| `max_edge_length` | `float \| None` | `None` | Remove triangles with edges longer than this |

Automatically extracts vertex colors from Red/Green/Blue dimensions if present.

### Mesh Class

```python
mesh.num_vertices    # int
mesh.num_faces       # int
mesh.vertices        # (N, 3) float64 array
mesh.faces           # (M, 3) int64 array
mesh.vertex_colors   # (N, 3) uint8 array or None
```

### Mesh Export

```python
# Write to specific format
mesh.write_obj("output.obj")    # Wavefront OBJ
mesh.write_stl("output.stl")    # Binary STL
mesh.write_ply("output.ply")    # ASCII PLY (with colors if available)

# Auto-detect from extension
mesh.write("output.obj")

# Decimate (reduce face count)
decimated = mesh.decimate(target_faces=10000)
decimated.write("simplified.obj")
```

---

## Parallel Processing (Dask)

Optional Dask integration for parallel chunk processing. Requires `pip install pywolken[dask]`.

**Module:** `pywolken.parallel`

```python
from pywolken.parallel import parallel_read, parallel_filter, parallel_apply

# Read multiple files in parallel
pc = parallel_read(["tile1.laz", "tile2.laz", "tile3.laz"])

# Apply per-point filters in parallel chunks
from pywolken.filters.range import RangeFilter
from pywolken.filters.expression import ExpressionFilter

result = parallel_filter(
    pc,
    filters=[
        RangeFilter(limits="Classification[2:2]"),
        ExpressionFilter(expression="Z > 100"),
    ],
    n_chunks=8,
)

# Apply custom function in parallel
result = parallel_apply(pc, lambda chunk: chunk.mask(chunk["Z"] > 100), n_chunks=4)
```

#### `parallel_read(paths, reader=None, **options) -> PointCloud`

Read and merge multiple files in parallel.

#### `parallel_filter(pc, filters, n_chunks=4) -> PointCloud`

Split point cloud into chunks, apply filters in parallel, merge results.

**Only for per-point filters** (range, expression, assign, decimation) — NOT for spatial filters (ground, cluster, normal, outlier).

#### `parallel_apply(pc, func, n_chunks=4) -> PointCloud`

Apply a custom `PointCloud -> PointCloud` function to chunks in parallel.

---

## CLI Reference

The `pywolken` command is installed as a console script entry point.

### `pywolken info`

Show point cloud file information.

```bash
pywolken info terrain.laz
```

Output:
```
File: terrain.laz
Size: 289.3 MB
Points: 45,266,951
Dimensions: X, Y, Z, Intensity, ReturnNumber, ...
CRS: EPSG:25832
Bounds X: [399000.000, 400000.000]
Bounds Y: [5600000.000, 5601000.000]
Bounds Z: [318.020, 502.340]
Format: las
Point format: 6
LAS version: 1.4
```

### `pywolken pipeline`

Run a JSON pipeline file.

```bash
pywolken pipeline workflow.json
pywolken pipeline workflow.json -v    # Verbose logging
```

### `pywolken convert`

Convert between point cloud formats.

```bash
pywolken convert input.laz output.ply
pywolken convert input.laz output.csv
pywolken convert input.ply output.las
```

### `pywolken merge`

Merge multiple files into one.

```bash
pywolken merge tile1.laz tile2.laz tile3.laz -o merged.laz
```

### `pywolken --version`

Print version and exit.

---

## Architecture

### Package Structure

```
src/pywolken/
├── __init__.py              # Public API exports
├── _version.py              # Version: "0.1.0"
├── cli.py                   # CLI entry point
├── parallel.py              # Optional Dask integration
│
├── core/                    # Core data model
│   ├── pointcloud.py        # PointCloud class (dict-of-arrays)
│   ├── dimensions.py        # Standard dimension definitions
│   ├── bounds.py            # Axis-aligned bounding box
│   └── metadata.py          # Source metadata container
│
├── io/                      # Readers & Writers
│   ├── base.py              # Reader/Writer abstract base classes
│   ├── registry.py          # Format auto-detection, read/write functions
│   ├── las.py               # LAS/LAZ via laspy[lazrs]
│   ├── ply.py               # PLY (ASCII + binary)
│   ├── csv.py               # CSV/TXT/XYZ (delimited text)
│   └── gdal.py              # GeoTIFF DEM writer via rasterio
│
├── filters/                 # 15 processing filters
│   ├── base.py              # Filter ABC
│   ├── registry.py          # Filter discovery & registration
│   ├── range.py             # Dimension range filtering
│   ├── crop.py              # Spatial bounding box crop
│   ├── merge.py             # Merge multiple clouds
│   ├── decimation.py        # Subsampling (step/fraction/count)
│   ├── assign.py            # Set dimension values
│   ├── expression.py        # Boolean expression filter
│   ├── reprojection.py      # CRS transformation
│   ├── ground.py            # SMRF ground classification
│   ├── hag.py               # Height Above Ground
│   ├── outlier.py           # Statistical/radius outlier removal
│   ├── normal.py            # PCA surface normals
│   ├── voxel.py             # Voxel grid downsampling
│   ├── cluster.py           # DBSCAN clustering
│   ├── colorize.py          # RGB from raster overlay
│   └── sort.py              # Morton/XYZ spatial sorting
│
├── pipeline/                # Pipeline engine
│   ├── pipeline.py          # JSON pipeline (PDAL-compatible)
│   └── streaming.py         # Memory-bounded chunked pipeline
│
├── raster/                  # Raster processing
│   ├── dem.py               # DEM generation (IDW/mean/nearest/TIN)
│   ├── hillshade.py         # Hillshade (Horn's method)
│   └── export.py            # GeoTIFF export via rasterio
│
├── mesh/                    # 3D mesh generation
│   └── triangulate.py       # 2.5D Delaunay, OBJ/STL/PLY export
│
└── utils/                   # Utilities
    └── crs.py               # CRS parsing and reprojection helpers
```

### Design Principles

1. **Columnar storage:** Each dimension is a separate NumPy array. Excellent cache locality for single-dimension operations (filtering, statistics). Easy to add/remove dimensions.

2. **Immutable filters:** Filters never modify the input PointCloud. They always return a new object.

3. **Plugin registries:** Readers, writers, and filters are registered by type name. Easy to extend with custom implementations.

4. **Lazy registration:** Built-in I/O handlers and filters are only imported when first used, keeping import time fast.

5. **PDAL compatibility:** JSON pipeline format, range syntax, filter naming conventions, and CLI patterns mirror PDAL for easy migration.

6. **No C++ compilation:** All dependencies ship as pre-built wheels (lazrs = Rust, NumPy/SciPy = C/Fortran, but all available via pip).

### Adding Custom Filters

```python
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry
from pywolken.core.pointcloud import PointCloud

class MyFilter(Filter):
    def filter(self, pc: PointCloud) -> PointCloud:
        threshold = float(self.options.get("threshold", 100))
        return pc.mask(pc["Z"] > threshold)

    @classmethod
    def type_name(cls) -> str:
        return "filters.my_custom"

# Register
filter_registry.register(MyFilter)

# Now usable in pipelines:
# {"type": "filters.my_custom", "threshold": 200}
```

### Adding Custom I/O Formats

```python
from pywolken.io.base import Reader, Writer
from pywolken.io.registry import io_registry

class MyReader(Reader):
    def read(self, path, **options):
        # Parse file → PointCloud
        ...

    @classmethod
    def extensions(cls):
        return [".myformat"]

    @classmethod
    def type_name(cls):
        return "readers.myformat"

io_registry.register_reader(MyReader)
```

---

## Dependencies

### Runtime Dependency Graph

```
pywolken
├── numpy >= 1.24           (array operations)
├── laspy[lazrs] >= 2.5     (LAS/LAZ I/O)
│   └── lazrs               (Rust LAZ codec, pre-built wheel)
├── scipy >= 1.10           (KDTree, Delaunay, interpolation)
└── pyproj >= 3.5           (CRS definitions, reprojection)

Optional:
├── rasterio >= 1.3         (GeoTIFF export, raster colorize)
├── dask[array] >= 2023.0   (parallel processing)
├── matplotlib >= 3.7       (2D visualization)
├── plotly >= 5.0           (3D visualization)
└── open3d >= 0.17          (advanced mesh processing)
```

---

## Examples

### Full Workflow: LAZ to Hillshade GeoTIFF

```python
import pywolken
from pywolken.filters.ground import GroundFilter
from pywolken.filters.range import RangeFilter
from pywolken.raster.dem import create_dem
from pywolken.raster.hillshade import hillshade
from pywolken.raster.export import write_geotiff

# 1. Read LAZ
pc = pywolken.read("terrain.laz")

# 2. Ground classification
pc = GroundFilter(cell_size=1.0, slope=0.15).filter(pc)

# 3. Extract ground points
ground = RangeFilter(limits="Classification[2:2]").filter(pc)

# 4. Generate DEM
raster, transform = create_dem(ground, resolution=0.5, method="idw")

# 5. Compute hillshade
hs = hillshade(raster, resolution=0.5, azimuth=315, altitude=45, z_factor=2.0)

# 6. Export GeoTIFF
write_geotiff(hs, "hillshade.tif", transform, nodata=None, dtype="uint8")
```

### JSON Pipeline: Ground Extraction

```json
{
  "pipeline": [
    "terrain.laz",
    {"type": "filters.ground", "cell_size": 1.0},
    {"type": "filters.range", "limits": "Classification[2:2]"},
    "ground_only.laz"
  ]
}
```

```bash
pywolken pipeline ground_pipeline.json -v
```

### Format Conversion

```python
import pywolken

# LAS → PLY
pc = pywolken.read("input.laz")
pywolken.write(pc, "output.ply", format="binary_little_endian")

# LAS → CSV
pywolken.write(pc, "output.csv", delimiter=",", precision=3)

# CSV → LAS
pc = pywolken.read("points.csv", header="X,Y,Z,Intensity")
pywolken.write(pc, "output.las")
```

### 3D Mesh from Point Cloud

```python
import pywolken
from pywolken.mesh import triangulate_2d

pc = pywolken.read("terrain.laz")
ground = pc.mask(pc["Classification"] == 2)

# Decimate for manageable mesh size
from pywolken.filters.voxel import VoxelFilter
ground_voxel = VoxelFilter(cell_size=2.0).filter(ground)

# Triangulate and export
mesh = triangulate_2d(ground_voxel, max_edge_length=10.0)
mesh.write("terrain.obj")
mesh.write("terrain.stl")
mesh.write("terrain.ply")

# Decimate mesh further
small = mesh.decimate(target_faces=50000)
small.write("terrain_simple.obj")
```

### Streaming Large Files

```python
from pywolken import StreamingPipeline
from pywolken.io.las import LasReader, LasWriter
from pywolken.filters.range import RangeFilter
from pywolken.filters.decimation import DecimationFilter

sp = StreamingPipeline(
    reader=LasReader(),
    input_path="huge_4gb.laz",
    writer=LasWriter(),
    output_path="filtered.laz",
    filters=[
        RangeFilter(limits="Classification[2:2]"),
        DecimationFilter(step=5),
    ],
    chunk_size=2_000_000,
)
total = sp.execute()
print(f"Read {sp.total_read:,} → Wrote {sp.total_written:,} points")
```

### Parallel Processing with Dask

```python
from pywolken.parallel import parallel_read, parallel_filter
from pywolken.filters.range import RangeFilter

# Read multiple tiles in parallel
pc = parallel_read([
    "tile_001.laz", "tile_002.laz", "tile_003.laz",
    "tile_004.laz", "tile_005.laz", "tile_006.laz",
])

# Filter in parallel chunks
result = parallel_filter(
    pc,
    filters=[RangeFilter(limits="Classification[2:6]")],
    n_chunks=8,
)
```

### Chained Filter Processing

```python
import pywolken
from pywolken.filters.ground import GroundFilter
from pywolken.filters.hag import HagFilter
from pywolken.filters.outlier import OutlierFilter
from pywolken.filters.expression import ExpressionFilter

pc = pywolken.read("terrain.laz")

# Chain: outlier removal → ground → HAG → filter tall objects
pc = OutlierFilter(method="statistical", mean_k=8, multiplier=2.0).filter(pc)
pc = GroundFilter().filter(pc)
pc = HagFilter().filter(pc)
tall = ExpressionFilter(expression="HeightAboveGround > 20").filter(pc)

print(f"Found {tall.num_points:,} points above 20m")
pywolken.write(tall, "tall_objects.laz")
```

### Point Cloud Statistics

```python
import pywolken
import numpy as np

pc = pywolken.read("terrain.laz")

print(f"Points:    {pc.num_points:,}")
print(f"CRS:       {pc.crs}")
print(f"Bounds:    {pc.bounds}")
print(f"Z range:   {pc['Z'].min():.2f} - {pc['Z'].max():.2f} m")
print(f"Mean Z:    {pc['Z'].mean():.2f} m")
print(f"Dims:      {pc.dimensions}")

# Classification breakdown
if "Classification" in pc:
    from pywolken.core.dimensions import CLASSIFICATION_CODES
    classes, counts = np.unique(pc["Classification"], return_counts=True)
    for cls, count in zip(classes, counts):
        name = CLASSIFICATION_CODES.get(int(cls), "Unknown")
        print(f"  Class {cls} ({name}): {count:,}")
```
