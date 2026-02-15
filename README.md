# pywolken

[![PyPI version](https://img.shields.io/pypi/v/pywolken)](https://pypi.org/project/pywolken/)
[![Python](https://img.shields.io/pypi/pyversions/pywolken)](https://pypi.org/project/pywolken/)
[![License: MIT](https://img.shields.io/pypi/l/pywolken)](https://github.com/jonash54/pywolken/blob/master/LICENSE)

Python point cloud processing library — the Python alternative to PDAL.

No C++ compilation required. Pure Python with NumPy, SciPy, laspy, pyproj.

## Installation

```bash
pip install pywolken
```

Optional extras:

```bash
pip install pywolken[raster]    # + GeoTIFF export (rasterio)
pip install pywolken[viz]       # + matplotlib, plotly
pip install pywolken[dask]      # + parallel processing
pip install pywolken[mesh]      # + 3D mesh (open3d)
pip install pywolken[all]       # everything
```

## Features

- **I/O:** LAS/LAZ, PLY (ASCII + binary), CSV/TXT/XYZ, GeoTIFF
- **15 Filters:** range, crop, merge, decimation, assign, expression, reprojection, ground classification (SMRF), height above ground, outlier removal, surface normals, voxel downsampling, DBSCAN clustering, raster colorization, spatial sorting
- **JSON Pipelines:** PDAL-compatible declarative processing chains
- **Raster:** DEM generation (IDW/mean/nearest/TIN), hillshade (Horn's method), GeoTIFF export
- **3D Mesh:** 2.5D Delaunay triangulation, OBJ/STL/PLY export
- **Streaming:** Memory-bounded chunked processing for huge files
- **Parallel:** Optional Dask integration for multi-file and multi-chunk processing
- **CLI:** `pywolken info`, `pywolken pipeline`, `pywolken convert`, `pywolken merge`

## Quick Start

```python
import pywolken

# Read any format (auto-detected)
pc = pywolken.read("terrain.laz")
print(pc)  # PointCloud(45,266,951 points, dims=[X, Y, Z, Intensity, ...])

# Filter
ground = pc.mask(pc["Classification"] == 2)

# Write to any format
pywolken.write(ground, "ground.laz")
pywolken.write(ground, "ground.ply")
pywolken.write(ground, "ground.csv")
```

## JSON Pipeline

```python
import json
pipeline = pywolken.Pipeline(json.dumps({
    "pipeline": [
        "input.laz",
        {"type": "filters.ground"},
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {"type": "filters.decimation", "step": 10},
        "output.las"
    ]
}))
count = pipeline.execute()
```

## CLI

```bash
pywolken info terrain.laz
pywolken convert input.laz output.ply
pywolken pipeline workflow.json -v
pywolken merge tile1.laz tile2.laz -o merged.laz
```

## Full Documentation

- [English](docs/DOCUMENTATION.md) — complete API reference, all filter options, examples, and architecture guide
- [Deutsch](docs/DOKUMENTATION.md) — vollständige API-Referenz, alle Filteroptionen, Beispiele und Architektur

## License

MIT
