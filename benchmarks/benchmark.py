"""Performance benchmark: pywolken vs PDAL on real LiDAR data."""

import json
import os
import sys
import time
import tempfile

import numpy as np

LAZ_FILE = "/var/geodata/lpolpg_32_419_5621_1_rp.laz"


def bench(name, func):
    """Run a benchmark and record the result."""
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0
    print(f"  {name}: {elapsed:.3f}s")
    return elapsed, result


def run_pywolken():
    print("\n" + "=" * 60)
    print("PYWOLKEN benchmarks")
    print("=" * 60)

    sys.path.insert(0, "src")
    import pywolken
    from pywolken.filters.registry import get_filter
    from pywolken.raster.dem import create_dem

    results = {}

    # 1. Read LAZ
    t, pc = bench("pw_read_laz", lambda: pywolken.read(LAZ_FILE))
    results["pw_read_laz"] = t
    print(f"    -> {pc.num_points:,} points, {len(pc.dimensions)} dims")

    # 2. Range filter — ground points (classification == 2)
    t, ground = bench("pw_filter_range",
        lambda: get_filter("filters.range", limits="Classification[2:2]").filter(pc))
    results["pw_filter_range"] = t
    print(f"    -> {ground.num_points:,} ground points")

    # 3. Decimation (keep every 10th point)
    t, dec = bench("pw_decimate",
        lambda: get_filter("filters.decimation", step=10).filter(pc))
    results["pw_decimate"] = t
    print(f"    -> {dec.num_points:,} decimated")

    # 4. Normal estimation (on 500k subset)
    subset = pc.slice(0, 500_000)
    t, _ = bench("pw_normals_500k",
        lambda: get_filter("filters.normal", k=8).filter(subset))
    results["pw_normals_500k"] = t

    # 5. DEM IDW (from first 1M ground points)
    ground_sub = ground.slice(0, min(1_000_000, ground.num_points))
    t, _ = bench("pw_dem_idw",
        lambda: create_dem(ground_sub, resolution=1.0, method="idw"))
    results["pw_dem_idw"] = t

    # 6. DEM mean
    t, _ = bench("pw_dem_mean",
        lambda: create_dem(ground_sub, resolution=1.0, method="mean"))
    results["pw_dem_mean"] = t

    # 7. Write LAS (5M points)
    subset_5m = pc.slice(0, 5_000_000)
    tmp_las = tempfile.mktemp(suffix=".las")
    t, _ = bench("pw_write_las_5M", lambda: pywolken.write(subset_5m, tmp_las))
    results["pw_write_las_5M"] = t
    os.unlink(tmp_las)

    # 8. Write LAZ (5M points)
    tmp_laz = tempfile.mktemp(suffix=".laz")
    t, _ = bench("pw_write_laz_5M", lambda: pywolken.write(subset_5m, tmp_laz))
    results["pw_write_laz_5M"] = t
    os.unlink(tmp_laz)

    # 9. Statistical outlier removal (500k)
    t, _ = bench("pw_outlier_500k",
        lambda: get_filter("filters.outlier", method="statistical", k=8, threshold=2.0).filter(subset))
    results["pw_outlier_500k"] = t

    # 10. Full pipeline (read -> range -> decimate -> write)
    tmp_pipe = tempfile.mktemp(suffix=".las")
    pipe_json = json.dumps({"pipeline": [
        LAZ_FILE,
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {"type": "filters.decimation", "step": 10},
        tmp_pipe,
    ]})
    t, _ = bench("pw_pipeline_full", lambda: pywolken.Pipeline(pipe_json).execute())
    results["pw_pipeline_full"] = t
    os.unlink(tmp_pipe)

    with open("/tmp/pw_bench.json", "w") as f:
        json.dump(results, f)
    return results


if __name__ == "__main__":
    run_pywolken()
