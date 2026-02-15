# pywolken

Python point cloud processing library — the Python alternative to PDAL.

```bash
pip install pywolken
```

## Quick Start

```python
import pywolken

# Read a point cloud
pc = pywolken.read("terrain.laz")
print(pc)  # PointCloud(12,345,678 points, dims=[X, Y, Z, Intensity, Classification, GpsTime])

# JSON pipeline (PDAL-compatible)
import json
pipeline = pywolken.Pipeline(json.dumps({
    "pipeline": [
        "input.laz",
        {"type": "filters.range", "limits": "Classification[2:2]"},
        "output.las"
    ]
}))
count = pipeline.execute()
```

## License

MIT
