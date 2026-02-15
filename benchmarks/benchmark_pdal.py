"""Performance benchmark: PDAL on real LiDAR data."""

import json
import os
import time
import tempfile

import numpy as np
import pdal

LAZ_FILE = "/var/geodata/lpolpg_32_419_5621_1_rp.laz"


def bench(name, func):
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0
    print(f"  {name}: {elapsed:.3f}s")
    return elapsed, result


def run_pdal():
    print("\n" + "=" * 60)
    print("PDAL benchmarks")
    print("=" * 60)

    results = {}

    # 1. Read LAZ
    def read_laz():
        p = pdal.Reader(LAZ_FILE).pipeline()
        p.execute()
        return p.arrays[0]

    t, arr = bench("pdal_read_laz", read_laz)
    results["pdal_read_laz"] = t
    print(f"    -> {len(arr):,} points")

    # 2. Range filter — ground points (includes read)
    def range_filter():
        p = pdal.Reader(LAZ_FILE) | pdal.Filter.range(limits="Classification[2:2]")
        p.execute()
        return p.arrays[0]

    t, ground_arr = bench("pdal_filter_range", range_filter)
    results["pdal_filter_range"] = t
    print(f"    -> {len(ground_arr):,} ground points")

    # 3. Decimation (includes read)
    def decimate():
        p = pdal.Reader(LAZ_FILE) | pdal.Filter.decimation(step=10)
        p.execute()
        return p.arrays[0]

    t, dec_arr = bench("pdal_decimate", decimate)
    results["pdal_decimate"] = t
    print(f"    -> {len(dec_arr):,} decimated")

    # 4. Normal estimation (500k subset)
    subset_500k = arr[:500_000]
    tmp_sub = tempfile.mktemp(suffix=".las")
    pw = pdal.Writer(tmp_sub).pipeline(subset_500k)
    pw.execute()

    def normals():
        p = pdal.Reader(tmp_sub) | pdal.Filter.normal(knn=8)
        p.execute()
        return p.arrays[0]

    t, _ = bench("pdal_normals_500k", normals)
    results["pdal_normals_500k"] = t
    os.unlink(tmp_sub)

    # 5. DEM IDW (1M ground points)
    ground_sub = ground_arr[:min(1_000_000, len(ground_arr))]
    tmp_ground = tempfile.mktemp(suffix=".las")
    pw = pdal.Writer(tmp_ground).pipeline(ground_sub)
    pw.execute()

    def dem_idw():
        tmp_tif = tempfile.mktemp(suffix=".tif")
        p = pdal.Reader(tmp_ground) | pdal.Writer.gdal(
            filename=tmp_tif, resolution=1.0, output_type="idw",
        )
        p.execute()
        os.unlink(tmp_tif)

    t, _ = bench("pdal_dem_idw", dem_idw)
    results["pdal_dem_idw"] = t

    # 6. DEM mean
    def dem_mean():
        tmp_tif = tempfile.mktemp(suffix=".tif")
        p = pdal.Reader(tmp_ground) | pdal.Writer.gdal(
            filename=tmp_tif, resolution=1.0, output_type="mean",
        )
        p.execute()
        os.unlink(tmp_tif)

    t, _ = bench("pdal_dem_mean", dem_mean)
    results["pdal_dem_mean"] = t
    os.unlink(tmp_ground)

    # 7. Write LAS (5M points)
    subset_5m = arr[:5_000_000]
    tmp_las = tempfile.mktemp(suffix=".las")

    def write_las():
        p = pdal.Writer(tmp_las).pipeline(subset_5m)
        p.execute()

    t, _ = bench("pdal_write_las_5M", write_las)
    results["pdal_write_las_5M"] = t
    os.unlink(tmp_las)

    # 8. Write LAZ (5M points)
    tmp_laz = tempfile.mktemp(suffix=".laz")

    def write_laz():
        p = pdal.Writer.las(tmp_laz, compression="lazperf").pipeline(subset_5m)
        p.execute()

    t, _ = bench("pdal_write_laz_5M", write_laz)
    results["pdal_write_laz_5M"] = t
    os.unlink(tmp_laz)

    # 9. Statistical outlier removal (500k)
    subset_500k = arr[:500_000]
    tmp_sub = tempfile.mktemp(suffix=".las")
    pw = pdal.Writer(tmp_sub).pipeline(subset_500k)
    pw.execute()

    def outlier():
        p = pdal.Reader(tmp_sub) | pdal.Filter.outlier(method="statistical", mean_k=8, multiplier=2.0)
        p.execute()
        return p.arrays[0]

    t, _ = bench("pdal_outlier_500k", outlier)
    results["pdal_outlier_500k"] = t
    os.unlink(tmp_sub)

    # 10. Full pipeline (read -> range -> decimate -> write)
    tmp_pipe = tempfile.mktemp(suffix=".las")

    def pipeline():
        p = (
            pdal.Reader(LAZ_FILE)
            | pdal.Filter.range(limits="Classification[2:2]")
            | pdal.Filter.decimation(step=10)
            | pdal.Writer(tmp_pipe)
        )
        p.execute()

    t, _ = bench("pdal_pipeline_full", pipeline)
    results["pdal_pipeline_full"] = t
    os.unlink(tmp_pipe)

    with open("/tmp/pdal_bench.json", "w") as f:
        json.dump(results, f)
    return results


if __name__ == "__main__":
    run_pdal()
