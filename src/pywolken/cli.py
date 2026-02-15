"""pywolken CLI — command-line interface for point cloud processing."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from pywolken._version import __version__


def cmd_info(args: argparse.Namespace) -> int:
    """Show info about a point cloud file."""
    from pywolken.io.registry import read

    path = args.file
    if not Path(path).exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    print(f"File: {path}")
    print(f"Size: {Path(path).stat().st_size / 1024 / 1024:.1f} MB")

    pc = read(path)
    print(f"Points: {pc.num_points:,}")
    print(f"Dimensions: {', '.join(pc.dimensions)}")

    if pc.crs:
        print(f"CRS: {pc.crs}")

    if "X" in pc and "Y" in pc and "Z" in pc:
        bounds = pc.bounds
        print(f"Bounds X: [{bounds.minx:.3f}, {bounds.maxx:.3f}]")
        print(f"Bounds Y: [{bounds.miny:.3f}, {bounds.maxy:.3f}]")
        print(f"Bounds Z: [{bounds.minz:.3f}, {bounds.maxz:.3f}]")

    if pc.metadata.source_format:
        print(f"Format: {pc.metadata.source_format}")
    if pc.metadata.point_format_id is not None:
        print(f"Point format: {pc.metadata.point_format_id}")
    if pc.metadata.file_version:
        print(f"LAS version: {pc.metadata.file_version}")

    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run a JSON pipeline."""
    from pywolken.pipeline import Pipeline

    pipeline_path = args.pipeline
    if not Path(pipeline_path).exists():
        print(f"Error: Pipeline file not found: {pipeline_path}", file=sys.stderr)
        return 1

    json_str = Path(pipeline_path).read_text()
    pipeline = Pipeline(json_str)

    errors = pipeline.validate()
    if errors:
        for e in errors:
            print(f"Validation error: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.time()
    count = pipeline.execute()
    elapsed = time.time() - t0

    print(f"Processed {count:,} points in {elapsed:.1f}s")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert between point cloud formats."""
    from pywolken.io.registry import read, write

    input_path = args.input
    output_path = args.output

    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.time()
    pc = read(input_path)
    n = write(pc, output_path)
    elapsed = time.time() - t0

    print(f"Converted {n:,} points: {input_path} -> {output_path} ({elapsed:.1f}s)")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge multiple point cloud files."""
    from pywolken.io.registry import read, write

    inputs = args.inputs
    output = args.output

    for p in inputs:
        if not Path(p).exists():
            print(f"Error: File not found: {p}", file=sys.stderr)
            return 1

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.time()
    result = read(inputs[0])
    for p in inputs[1:]:
        other = read(p)
        result = result.merge(other)

    n = write(result, output)
    elapsed = time.time() - t0
    print(f"Merged {len(inputs)} files -> {n:,} points ({elapsed:.1f}s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pywolken",
        description="pywolken — Python point cloud processing library",
    )
    parser.add_argument(
        "--version", action="version", version=f"pywolken {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    info_parser = subparsers.add_parser("info", help="Show point cloud file info")
    info_parser.add_argument("file", help="Point cloud file path")

    # pipeline
    pipe_parser = subparsers.add_parser("pipeline", help="Run a JSON pipeline")
    pipe_parser.add_argument("pipeline", help="Path to JSON pipeline file")
    pipe_parser.add_argument("-v", "--verbose", action="store_true")

    # convert
    conv_parser = subparsers.add_parser("convert", help="Convert between formats")
    conv_parser.add_argument("input", help="Input file path")
    conv_parser.add_argument("output", help="Output file path")
    conv_parser.add_argument("-v", "--verbose", action="store_true")

    # merge
    merge_parser = subparsers.add_parser("merge", help="Merge multiple files")
    merge_parser.add_argument("inputs", nargs="+", help="Input file paths")
    merge_parser.add_argument("-o", "--output", required=True, help="Output file path")
    merge_parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "info": cmd_info,
        "pipeline": cmd_pipeline,
        "convert": cmd_convert,
        "merge": cmd_merge,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
