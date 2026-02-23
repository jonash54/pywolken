# Changelog

## v0.1.1 (2026-02-22)

### Bugfixes
- **Hillshade:** Fix inverted relief — corrected aspect angle from `arctan2(-dzdy, dzdx)` to `arctan2(dzdy, -dzdx)` so lit/shadowed faces render correctly
- **Hillshade:** Fix steep slope over-brightening — add normalization denominator `sqrt(1 + slope²)` to the shading formula

## v0.1.0 (2026-02-15)

Initial release.

### Features
- **I/O:** LAS/LAZ (via laspy/lazrs), PLY (ASCII + binary), CSV/TXT/XYZ, GeoTIFF
- **15 Filters:** range, crop, merge, decimation, assign, expression, reprojection, ground classification (SMRF), height above ground, outlier removal, surface normals, voxel downsampling, DBSCAN clustering, raster colorization, spatial sorting
- **JSON Pipelines:** PDAL-compatible declarative processing chains
- **Raster:** DEM generation (IDW/mean/nearest/TIN), hillshade (Horn's method), GeoTIFF export
- **3D Mesh:** 2.5D Delaunay triangulation, OBJ/STL/PLY export
- **Streaming:** Memory-bounded chunked processing for huge files
- **Parallel:** Optional Dask integration for multi-file and multi-chunk processing
- **CLI:** `pywolken info`, `pywolken pipeline`, `pywolken convert`, `pywolken merge`
