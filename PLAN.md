# pywolken — Architecture & Implementation Plan

## Vision
A complete Python point cloud processing library — the Python alternative to PDAL.
Easy to install (`pip install pywolken`), capable of handling billions of points,
with a JSON pipeline system for reproducible workflows.

---

## 1. Package Structure

```
pywolken/
├── src/
│   └── pywolken/
│       ├── __init__.py                 # Public API, version
│       ├── _version.py                 # Version string
│       │
│       ├── core/                       # Core data model
│       │   ├── __init__.py
│       │   ├── pointcloud.py           # PointCloud class (dict-of-arrays)
│       │   ├── dimensions.py           # Dimension registry & standard dims
│       │   ├── bounds.py               # BBox / spatial bounds
│       │   └── metadata.py             # Metadata container
│       │
│       ├── io/                         # Readers & Writers
│       │   ├── __init__.py
│       │   ├── base.py                 # Reader/Writer ABC
│       │   ├── registry.py             # Format auto-detection & plugin registry
│       │   ├── las.py                  # LAS/LAZ via laspy[lazrs]
│       │   ├── ply.py                  # PLY (ASCII + binary)
│       │   └── csv.py                  # CSV/text delimited
│       │
│       ├── filters/                    # Processing filters
│       │   ├── __init__.py
│       │   ├── base.py                 # Filter ABC with streaming support
│       │   ├── registry.py             # Filter discovery & registration
│       │   ├── range.py                # Dimension range filtering
│       │   ├── crop.py                 # Spatial bounding box crop
│       │   ├── merge.py                # Merge multiple point clouds
│       │   ├── decimation.py           # Nth-point / random / voxel decimation
│       │   ├── assign.py              # Assign values to dimensions
│       │   ├── expression.py           # Filter by expression
│       │   ├── reprojection.py         # CRS transformation (pyproj)
│       │   ├── outlier.py              # Statistical & radius outlier removal
│       │   ├── ground.py              # Ground classification (SMRF/PMF/CSF)
│       │   ├── hag.py                  # Height Above Ground
│       │   ├── normal.py               # Surface normal estimation
│       │   ├── voxel.py                # Voxel downsampling
│       │   ├── cluster.py              # DBSCAN / Euclidean clustering
│       │   ├── colorize.py             # RGB from raster overlay
│       │   └── sort.py                 # Spatial sorting (Morton code)
│       │
│       ├── pipeline/                   # JSON Pipeline Engine
│       │   ├── __init__.py
│       │   ├── pipeline.py             # Pipeline class — parse, validate, execute
│       │   ├── schema.py               # JSON schema for pipeline validation
│       │   └── streaming.py            # Chunked/streaming pipeline executor
│       │
│       ├── raster/                     # DEM & Hillshade
│       │   ├── __init__.py
│       │   ├── dem.py                  # DEM/DTM/DSM generation (IDW, TIN, nearest)
│       │   ├── hillshade.py            # Hillshade computation
│       │   └── export.py               # GeoTIFF export (rasterio)
│       │
│       ├── mesh/                       # 3D Mesh (Phase 6)
│       │   ├── __init__.py
│       │   ├── delaunay.py             # 2.5D Delaunay triangulation
│       │   ├── poisson.py              # Poisson surface reconstruction
│       │   └── export.py               # OBJ/STL/PLY mesh export
│       │
│       └── utils/                      # Shared utilities
│           ├── __init__.py
│           ├── spatial.py              # KDTree wrappers, spatial queries
│           ├── crs.py                  # CRS helpers (pyproj wrappers)
│           └── parallel.py             # Chunked processing & optional Dask
│
├── tests/
│   ├── conftest.py                     # Fixtures, small test point clouds
│   ├── data/                           # Small test LAS/PLY/CSV files
│   ├── test_core/
│   │   ├── test_pointcloud.py
│   │   └── test_dimensions.py
│   ├── test_io/
│   │   ├── test_las.py
│   │   ├── test_ply.py
│   │   └── test_csv.py
│   ├── test_filters/
│   │   ├── test_range.py
│   │   ├── test_crop.py
│   │   ├── test_decimation.py
│   │   └── ...
│   ├── test_pipeline/
│   │   └── test_pipeline.py
│   └── test_raster/
│       ├── test_dem.py
│       └── test_hillshade.py
│
├── pyproject.toml
├── README.md
├── LICENSE                             # MIT
└── .github/
    └── workflows/
        └── ci.yml                      # Tests + lint on push
```

---

## 2. Core Data Model

### 2.1 PointCloud (dict-of-arrays)

```python
class PointCloud:
    """Core point cloud container — columnar storage (dict of NumPy arrays)."""

    def __init__(self):
        self._arrays: dict[str, np.ndarray] = {}  # "X" -> float64[], etc.
        self._metadata: Metadata = Metadata()
        self._crs: CRS | None = None

    # Properties
    @property
    def num_points(self) -> int: ...
    @property
    def dimensions(self) -> list[str]: ...
    @property
    def bounds(self) -> Bounds: ...
    @property
    def crs(self) -> CRS | None: ...

    # Array access (NumPy-native)
    def __getitem__(self, key: str) -> np.ndarray: ...        # pc["X"]
    def __setitem__(self, key: str, val: np.ndarray): ...     # pc["X"] = arr
    def __contains__(self, key: str) -> bool: ...             # "X" in pc
    def __len__(self) -> int: ...                             # len(pc)

    # Slicing
    def mask(self, mask: np.ndarray) -> "PointCloud": ...     # Boolean mask
    def slice(self, start: int, end: int) -> "PointCloud": ...

    # Conversion
    def to_numpy(self) -> np.ndarray: ...                     # Structured array
    def to_dict(self) -> dict[str, np.ndarray]: ...
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "PointCloud": ...
    @classmethod
    def from_dict(cls, d: dict[str, np.ndarray]) -> "PointCloud": ...

    # Utilities
    def add_dimension(self, name: str, dtype: np.dtype, fill: float = 0.0): ...
    def remove_dimension(self, name: str): ...
    def copy(self) -> "PointCloud": ...
    def merge(self, other: "PointCloud") -> "PointCloud": ...
```

**Why dict-of-arrays (columnar)?**
- Better cache locality when filtering on single dimensions (e.g., Classification)
- Add/remove dimensions without reallocating all memory
- Natural fit for NumPy vectorized operations
- Compatible with Dask arrays for distributed processing
- Similar to Apache Arrow / Parquet columnar model

### 2.2 Dimensions

```python
# Standard LAS dimensions with types
STANDARD_DIMENSIONS = {
    "X": np.float64,
    "Y": np.float64,
    "Z": np.float64,
    "Intensity": np.uint16,
    "ReturnNumber": np.uint8,
    "NumberOfReturns": np.uint8,
    "Classification": np.uint8,
    "ScanAngleRank": np.int8,
    "UserData": np.uint8,
    "PointSourceId": np.uint16,
    "GpsTime": np.float64,
    "Red": np.uint16,
    "Green": np.uint16,
    "Blue": np.uint16,
    "NIR": np.uint16,
}
```

### 2.3 Metadata & Bounds

```python
class Metadata:
    """Point cloud metadata container."""
    source_file: str | None
    source_format: str | None
    creation_date: datetime | None
    software: str                       # "pywolken X.Y.Z"
    extra: dict[str, Any]               # Free-form metadata

class Bounds:
    """3D axis-aligned bounding box."""
    minx: float; miny: float; minz: float
    maxx: float; maxy: float; maxz: float
```

---

## 3. I/O Architecture

### 3.1 Base Classes

```python
class Reader(ABC):
    """Base class for all readers."""

    @abstractmethod
    def read(self, path: str, **options) -> PointCloud: ...

    def read_chunked(self, path: str, chunk_size: int = 1_000_000,
                     **options) -> Iterator[PointCloud]:
        """Default: read all, then chunk. Subclasses can override for streaming."""
        pc = self.read(path, **options)
        for i in range(0, len(pc), chunk_size):
            yield pc.slice(i, i + chunk_size)

    @classmethod
    def extensions(cls) -> list[str]: ...          # [".las", ".laz"]
    @classmethod
    def type_name(cls) -> str: ...                 # "readers.las"


class Writer(ABC):
    """Base class for all writers."""

    @abstractmethod
    def write(self, pc: PointCloud, path: str, **options) -> int: ...

    @classmethod
    def extensions(cls) -> list[str]: ...
    @classmethod
    def type_name(cls) -> str: ...                 # "writers.las"
```

### 3.2 Format Registry

```python
class IORegistry:
    """Auto-detect format from extension, register plugins."""
    _readers: dict[str, type[Reader]]   # ".las" -> LasReader
    _writers: dict[str, type[Writer]]

    def get_reader(self, path: str) -> Reader: ...
    def get_writer(self, path: str) -> Writer: ...
    def register_reader(self, cls: type[Reader]): ...
    def register_writer(self, cls: type[Writer]): ...
```

### 3.3 LAS/LAZ Reader (via laspy)

```python
class LasReader(Reader):
    """LAS/LAZ reader using laspy[lazrs]."""

    def read(self, path, **options) -> PointCloud:
        # laspy.read() -> structured NumPy -> dict-of-arrays -> PointCloud
        ...

    def read_chunked(self, path, chunk_size=1_000_000, **options):
        # Uses laspy.open() context manager for true streaming
        with laspy.open(path) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                yield _laspy_chunk_to_pointcloud(chunk)

    @classmethod
    def extensions(cls): return [".las", ".laz"]

    @classmethod
    def type_name(cls): return "readers.las"
```

### 3.4 PLY Reader/Writer

- Parse ASCII and binary (little/big endian) PLY
- Map PLY properties to pywolken dimensions (x→X, y→Y, z→Z, etc.)
- Use numpy for efficient binary parsing

### 3.5 CSV Reader/Writer

- Configurable delimiter, header row, column mapping
- Use numpy.loadtxt or pandas.read_csv for speed
- Map columns to dimensions via options

---

## 4. Filter Architecture

### 4.1 Base Class

```python
class Filter(ABC):
    """Base class for all processing filters."""

    def __init__(self, **options):
        self.options = options

    @abstractmethod
    def filter(self, pc: PointCloud) -> PointCloud: ...

    @classmethod
    def type_name(cls) -> str: ...   # "filters.range"

    def validate_options(self): ...   # Raise on invalid config
```

### 4.2 Filter Priority & Phase Assignment

| Filter | Phase | Description |
|--------|-------|-------------|
| `filters.range` | 1 | Filter by dimension value ranges (e.g., Classification[2:2]) |
| `filters.merge` | 1 | Merge multiple point clouds |
| `filters.crop` | 2 | Crop by 2D/3D bounding box |
| `filters.decimation` | 2 | Nth-point, random, or voxel thinning |
| `filters.assign` | 2 | Set dimension values |
| `filters.expression` | 2 | Filter by arbitrary expression |
| `filters.reprojection` | 3 | CRS transformation via pyproj |
| `filters.outlier` | 4 | Statistical & radius outlier removal |
| `filters.ground` | 3 | Ground classification (SMRF/PMF) |
| `filters.hag` | 3 | Height Above Ground |
| `filters.normal` | 4 | Surface normal estimation |
| `filters.voxel` | 4 | Voxel grid downsampling |
| `filters.cluster` | 4 | DBSCAN / Euclidean clustering |
| `filters.colorize` | 4 | Assign RGB from raster overlay |
| `filters.sort` | 5 | Spatial sorting (Morton/Hilbert) |

### 4.3 Key Filter Designs

**Range filter** (Phase 1 — critical for ground point extraction):
```python
class RangeFilter(Filter):
    """Filter points by dimension value ranges.
    PDAL-compatible syntax: 'Classification[2:2]', 'Z[100:200]'
    """
    def filter(self, pc: PointCloud) -> PointCloud:
        mask = parse_and_apply_ranges(pc, self.options["limits"])
        return pc.mask(mask)
```

**Ground classification** (Phase 3 — SMRF algorithm):
```python
class GroundFilter(Filter):
    """Classify ground points using SMRF (Simple Morphological Filter).
    Sets Classification=2 for ground points.

    Options:
        cell_size: float = 1.0      # Grid cell size (meters)
        slope: float = 0.15         # Max terrain slope
        window_max: float = 18.0    # Max morphological window size
        threshold: float = 0.5      # Height threshold
    """
```

---

## 5. Pipeline Engine

### 5.1 JSON Pipeline Format

PDAL-compatible where possible:

```json
{
    "pipeline": [
        {
            "type": "readers.las",
            "filename": "input.laz"
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"
        },
        {
            "type": "filters.decimation",
            "step": 10
        },
        {
            "type": "writers.las",
            "filename": "output.laz"
        }
    ]
}
```

Shortcuts (like PDAL):
```json
{
    "pipeline": [
        "input.laz",
        { "type": "filters.range", "limits": "Classification[2:2]" },
        "output.laz"
    ]
}
```

### 5.2 Pipeline Executor

```python
class Pipeline:
    """JSON pipeline executor."""

    def __init__(self, json_str: str | None = None, stages: list | None = None):
        self.stages: list[Reader | Filter | Writer] = []
        if json_str:
            self._parse_json(json_str)
        elif stages:
            self.stages = stages

    def validate(self) -> list[str]:
        """Validate pipeline — return list of errors (empty = valid)."""
        ...

    def execute(self) -> int:
        """Execute pipeline, return number of points processed."""
        # 1. Read from reader(s)
        # 2. Apply filters in sequence
        # 3. Write to writer(s)
        # 4. Return point count
        ...

    def execute_streaming(self, chunk_size: int = 1_000_000) -> int:
        """Execute pipeline in chunks for large files."""
        ...

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Access result point data after execution."""
        ...

    @property
    def metadata(self) -> dict:
        """Pipeline execution metadata."""
        ...

    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, path: str) -> "Pipeline": ...
```

### 5.3 Streaming Executor

For billion-point datasets:
1. Reader produces chunks via `read_chunked()`
2. Each filter processes one chunk at a time
3. Writer appends chunks to output
4. Memory stays bounded regardless of input size

Filters that need global context (e.g., statistical outlier needing global mean):
- Two-pass approach: first pass computes statistics, second pass applies
- Or: estimate from sample, then apply

---

## 6. Raster / DEM Module

This is critical — it's the user's primary use case.

### 6.1 DEM Generation

```python
def create_dem(pc: PointCloud, resolution: float,
               method: str = "idw",        # "idw", "tin", "nearest", "mean"
               bounds: Bounds | None = None,
               window_size: int = 6,       # For void filling
               power: float = 2.0,         # IDW power parameter
               ) -> tuple[np.ndarray, dict]:
    """
    Generate DEM from point cloud (typically ground-classified points).

    Returns:
        (raster_array, transform_dict)  — 2D float32 array + geotransform
    """
```

Interpolation methods:
- **IDW** (Inverse Distance Weighting) — default, matches PDAL's writers.gdal
- **TIN** (Triangulated Irregular Network) — via scipy.spatial.Delaunay
- **Nearest neighbor** — scipy.interpolate.NearestNDInterpolator
- **Mean** — simple grid averaging

### 6.2 Hillshade

```python
def hillshade(dem: np.ndarray, resolution: float,
              azimuth: float = 315.0,
              altitude: float = 45.0,
              z_factor: float = 1.0
              ) -> np.ndarray:
    """
    Compute hillshade from DEM using Horn's method.
    Pure NumPy — no GDAL dependency needed.
    Returns uint8 array (0-255).
    """
    # Compute slope and aspect via np.gradient
    # Apply illumination model
```

### 6.3 GeoTIFF Export

```python
def write_geotiff(array: np.ndarray, path: str,
                  transform: Affine, crs: CRS,
                  nodata: float = -9999.0,
                  compress: str = "lzw"):
    """Write raster to GeoTIFF via rasterio."""
```

### 6.4 GDAL Writer (Pipeline Integration)

```python
class GdalWriter(Writer):
    """Write point cloud to raster (DEM) — replaces PDAL's writers.gdal.

    Options:
        resolution: float
        output_type: str = "idw"     # idw, tin, nearest, mean
        window_size: int = 6
        gdaldriver: str = "GTiff"
        gdalopts: str = "COMPRESS=LZW,TILED=YES"
    """
    @classmethod
    def type_name(cls): return "writers.gdal"
```

---

## 7. 3D Mesh Module (Phase 6)

### 7.1 Delaunay 2.5D Triangulation
- scipy.spatial.Delaunay on X,Y coordinates
- Z values preserved as vertex heights
- Export as OBJ/STL/PLY mesh

### 7.2 Poisson Surface Reconstruction
- Requires surface normals (from normal filter)
- Implement simplified Poisson or wrap Open3D (optional dep)

---

## 8. Dependencies

### Required (core)
```
numpy >= 1.24
laspy[lazrs] >= 2.5      # LAS/LAZ with Rust LAZ backend
```

### Required (geospatial)
```
scipy >= 1.10             # KDTree, Delaunay, interpolation
pyproj >= 3.5             # CRS / coordinate transforms
```

### Optional
```
rasterio >= 1.3           # GeoTIFF I/O (for DEM/hillshade export)
dask[array] >= 2023.0     # Distributed / chunked processing
matplotlib >= 3.7         # 2D visualization
plotly >= 5.0             # 3D interactive visualization
```

---

## 9. Phased Implementation Roadmap

### Phase 1: Foundation (Sessions 1-8)
**Goal:** Project skeleton, core data model, LAS/LAZ read/write, first working pipeline.

- [ ] Project setup: pyproject.toml, src layout, git init, venv
- [ ] Core: PointCloud class with dict-of-arrays storage
- [ ] Core: Dimensions registry, Bounds, Metadata
- [ ] I/O: Reader/Writer base classes and IORegistry
- [ ] I/O: LasReader — read LAS/LAZ via laspy into PointCloud
- [ ] I/O: LasWriter — write PointCloud to LAS/LAZ
- [ ] Pipeline: Basic Pipeline class — parse JSON, execute reader→writer
- [ ] Tests: PointCloud, LAS read/write, simple pipeline
- [ ] Verify: Read one of your real .laz files, inspect points, write back

**Milestone:** `pywolken` can read a .laz file and write it back out.

### Phase 2: Pipeline Engine & Basic Filters (Sessions 9-16)
**Goal:** Full pipeline engine with essential filters.

- [ ] Pipeline: Full JSON parsing with shortcuts (string = filename)
- [ ] Pipeline: Validation, error messages, metadata tracking
- [ ] Filter: `filters.range` — dimension range filtering (Classification[2:2])
- [ ] Filter: `filters.crop` — spatial bounding box crop
- [ ] Filter: `filters.merge` — combine multiple point clouds
- [ ] Filter: `filters.decimation` — nth-point / random thinning
- [ ] Filter: `filters.assign` — set dimension values
- [ ] Filter: `filters.expression` — filter by arbitrary expression
- [ ] Filter: Registry with auto-discovery
- [ ] Tests: Each filter, pipeline with multiple stages

**Milestone:** Can run PDAL-style JSON pipelines with basic filters.

### Phase 3: Geospatial — DEM & Hillshade (Sessions 17-24)
**Goal:** Replace your laz_hillshade.py workflow entirely.

- [ ] CRS: Coordinate reference system on PointCloud (from LAS header)
- [ ] Filter: `filters.reprojection` — CRS transform via pyproj
- [ ] Raster: DEM generation (IDW interpolation on grid)
- [ ] Raster: TIN interpolation (scipy.spatial.Delaunay)
- [ ] Raster: Hillshade computation (Horn's method, pure NumPy)
- [ ] Raster: GeoTIFF export via rasterio
- [ ] Writer: `writers.gdal` — pipeline stage for DEM output
- [ ] Filter: `filters.ground` — SMRF ground classification
- [ ] Filter: `filters.hag` — height above ground
- [ ] Tests: DEM, hillshade, compare output vs GDAL/PDAL
- [ ] Integration test: full LAZ → ground filter → DEM → hillshade → GeoTIFF

**Milestone:** `pywolken` can produce the same hillshades as your existing script,
without PDAL or gdaldem installed.

### Phase 4: Advanced Filters (Sessions 25-36)
**Goal:** Comprehensive filter library for real-world processing.

- [ ] Filter: `filters.outlier` — statistical + radius outlier removal
- [ ] Filter: `filters.normal` — surface normal estimation via KDTree
- [ ] Filter: `filters.voxel` — voxel grid downsampling
- [ ] Filter: `filters.cluster` — DBSCAN clustering
- [ ] Filter: `filters.colorize` — assign RGB from raster (rasterio)
- [ ] Filter: `filters.sort` — Morton/Hilbert spatial sorting
- [ ] Utils: KDTree wrapper (scipy.spatial.cKDTree)
- [ ] Improved ground classification (PMF variant)
- [ ] Tests: all new filters with synthetic + real data

**Milestone:** Feature parity with PDAL's most-used filters.

### Phase 5: Scale & Additional Formats (Sessions 37-48)
**Goal:** Handle billion-point datasets, more I/O formats.

- [ ] Streaming pipeline executor (chunked read → filter → write)
- [ ] Memory-bounded processing for arbitrarily large files
- [ ] I/O: PLY reader/writer (ASCII + binary)
- [ ] I/O: CSV reader/writer (configurable columns)
- [ ] Optional Dask integration for parallel/distributed processing
- [ ] Performance benchmarks vs PDAL
- [ ] Large file integration tests

**Milestone:** Can process your full 4.2GB dataset without running out of memory.

### Phase 6: 3D, CLI & Distribution (Sessions 49-60+)
**Goal:** Mesh generation, command-line tool, publish to PyPI/conda.

- [ ] Mesh: 2.5D Delaunay triangulation → OBJ/STL/PLY
- [ ] Mesh: Optional Poisson reconstruction (Open3D dep)
- [ ] CLI: `pywolken pipeline run pipeline.json`
- [ ] CLI: `pywolken info file.laz` (quick point cloud summary)
- [ ] Packaging: pyproject.toml finalized, build & upload to PyPI
- [ ] Packaging: conda-forge recipe
- [ ] Documentation: README, API docs (MkDocs or Sphinx)
- [ ] CI/CD: GitHub Actions (test on Linux/macOS/Windows)
- [ ] Visualization helpers (matplotlib 2D, plotly 3D)

**Milestone:** `pip install pywolken` works. Library on GitHub with docs and CI.

---

## 10. Testing Strategy

- **pytest** as test runner
- **Small synthetic point clouds** for unit tests (generated in conftest.py)
- **Integration tests** against real .laz files (from /var/geodata/)
- **Benchmark tests** comparing against PDAL output for correctness
- **Property-based tests** (hypothesis) for core data model
- Test coverage target: >80%

---

## 11. pyproject.toml (Draft)

```toml
[project]
name = "pywolken"
version = "0.1.0"
description = "Python point cloud processing library — the Python alternative to PDAL"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "Jonas" }]
readme = "README.md"
keywords = ["point-cloud", "lidar", "las", "laz", "pdal", "geospatial", "dem"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: GIS",
]

dependencies = [
    "numpy>=1.24",
    "laspy[lazrs]>=2.5",
    "scipy>=1.10",
    "pyproj>=3.5",
]

[project.optional-dependencies]
raster = ["rasterio>=1.3"]
viz = ["matplotlib>=3.7", "plotly>=5.0"]
dask = ["dask[array]>=2023.0"]
mesh = ["open3d>=0.17"]
all = ["pywolken[raster,viz,dask,mesh]"]
dev = ["pytest>=7.0", "pytest-cov", "ruff"]

[project.scripts]
pywolken = "pywolken.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pywolken"]

[tool.ruff]
line-length = 99

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## 12. What Gets Built First (Phase 1 Detail)

In the very first session after plan approval, we will:

1. **`git init`** + create project skeleton (all directories, empty __init__.py files)
2. **`pyproject.toml`** with dependencies
3. **`src/pywolken/core/pointcloud.py`** — the PointCloud class
4. **`src/pywolken/core/dimensions.py`** — standard dimension definitions
5. **`src/pywolken/core/bounds.py`** — Bounds class
6. **`src/pywolken/core/metadata.py`** — Metadata class
7. **`src/pywolken/io/base.py`** — Reader/Writer ABCs
8. **`src/pywolken/io/registry.py`** — IORegistry
9. **`src/pywolken/io/las.py`** — LasReader + LasWriter
10. **`src/pywolken/pipeline/pipeline.py`** — Basic Pipeline
11. **Tests** for all of the above
12. **Verify** by reading one of your real .laz tiles

After Phase 1, this should work:

```python
import pywolken
import json

# Direct API
pc = pywolken.read("terrain.laz")
print(f"{len(pc)} points, dims: {pc.dimensions}")
print(f"Z range: {pc['Z'].min():.1f} - {pc['Z'].max():.1f}")
pywolken.write(pc, "copy.las")

# JSON pipeline
pipeline = pywolken.Pipeline(json.dumps({
    "pipeline": [
        "terrain.laz",
        "output.las"
    ]
}))
count = pipeline.execute()
print(f"Processed {count} points")
```
