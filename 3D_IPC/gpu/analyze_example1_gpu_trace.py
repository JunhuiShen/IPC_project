#!/usr/bin/env python3
"""Read-only Nsight Systems SQLite analysis; trace durations are not solver time."""
import argparse
import json
from pathlib import Path
import sqlite3


def identifier(name):
    return '"' + name.replace('"', '""') + '"'


def interval_metrics(intervals):
    ordered = sorted(intervals)
    total = sum(end - start for start, end in ordered)
    union = 0
    if ordered:
        lo, hi = ordered[0]
        for start, end in ordered[1:]:
            if start > hi:
                union += hi - lo
                lo, hi = start, end
            else:
                hi = max(hi, end)
        union += hi - lo
    span = max((end for _, end in ordered), default=0) - min((start for start, _ in ordered), default=0)
    return {"count": len(ordered), "duration_sum_ns": total, "duration_sum_s": total / 1e9,
            "interval_union_ns": union, "interval_union_s": union / 1e9,
            "activity_span_ns": span, "activity_span_s": span / 1e9}


def analyze(trace):
    trace = Path(trace).resolve(strict=True)
    with sqlite3.connect(trace.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0]: [field[1] for field in db.execute("PRAGMA table_info(" + identifier(row[0]) + ")")]
                  for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings = {}
        if "StringIds" in tables:
            columns = {column.lower(): column for column in tables["StringIds"]}
            if "id" in columns and "value" in columns:
                strings = dict(db.execute("SELECT " + identifier(columns["id"]) + "," +
                                          identifier(columns["value"]) + ' FROM "StringIds"'))
        gpu_metadata = []
        if "TARGET_INFO_GPU" in tables:
            gpu_metadata = [{key: value.hex() if isinstance(value, bytes) else value
                             for key, value in dict(row).items()}
                            for row in db.execute('SELECT * FROM "TARGET_INFO_GPU"')]
        all_intervals = []

        def summarize(table, kernel=False):
            if table not in tables:
                if kernel:
                    raise ValueError("Missing CUPTI_ACTIVITY_KIND_KERNEL table")
                return {"present": False, **interval_metrics([]), "groups": []}
            columns = {column.lower(): column for column in tables[table]}
            if not {"start", "end"} <= columns.keys():
                raise ValueError(table + " lacks start/end timestamps")
            groups, intervals = {}, []
            execution = {kind: [] for kind in ("serial_1x1", "multithread", "unknown_geometry")}
            resources = {"registers_per_thread": ("registersperthread", "registerperthread"),
                         "local_bytes_per_thread": ("localmemoryperthread", "localbytesperthread"),
                         "local_bytes_total": ("localmemorytotal", "localbytestotal"),
                         "static_shared_bytes": ("staticsharedmemory",),
                         "dynamic_shared_bytes": ("dynamicsharedmemory",)}
            for index, row in enumerate(db.execute("SELECT * FROM " + identifier(table)), 1):
                def value(*names):
                    for name in names:
                        if name in columns and row[columns[name]] is not None:
                            return row[columns[name]]
                    return None
                start, end = value("start"), value("end")
                if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
                    raise ValueError(f"{table} row {index}: unfinished/invalid integer-ns interval {start!r}..{end!r}")
                intervals.append((start, end))
                name = value("demangledname", "shortname", "nameid", "name")
                if name is not None:
                    name = str(strings.get(name, f"StringIds:{name}")) if isinstance(name, int) else str(name)
                else:
                    name = ", ".join(f"{key}={value(key)}" for key in ("copykind", "srckind", "dstkind")
                                     if key in columns) or table
                group = groups.setdefault(name, {"name": name, "count": 0, "duration_sum_ns": 0,
                                                  "bytes_total": 0, "ranges": {}, "invalid_metadata": {}})
                group["count"] += 1
                group["duration_sum_ns"] += end - start
                byte_count = value("bytes")
                if isinstance(byte_count, int):
                    group["bytes_total"] += byte_count
                if kernel:
                    geometry = {}
                    for prefix in ("grid", "block"):
                        dimensions = [value(prefix + axis) for axis in "xyz"]
                        geometry[prefix] = None
                        if all(isinstance(d, int) and d > 0 for d in dimensions):
                            geometry[prefix] = dimensions[0] * dimensions[1] * dimensions[2]
                    measurements = {key: value(*aliases) for key, aliases in resources.items()}
                    measurements.update(grid_blocks=geometry["grid"], threads_per_block=geometry["block"],
                                        launch_threads=(geometry["grid"] * geometry["block"]
                                                        if all(geometry.values()) else None))
                    for key, number in measurements.items():
                        if number is not None:
                            if not isinstance(number, int) or number < 0:
                                # Some Nsight exports contain negative resource
                                # values. Preserve the evidence, but do not guess
                                # whether it is a sentinel or integer overflow.
                                warning = group["invalid_metadata"].setdefault(key, {"count": 0, "raw_values": []})
                                warning["count"] += 1
                                for raw in warning["raw_values"]:
                                    if raw["value"] == number:
                                        raw["count"] += 1
                                        break
                                else:
                                    warning["raw_values"].append({"value": number, "count": 1})
                                continue
                            bounds = group["ranges"].setdefault(key, {"min": number, "max": number})
                            bounds.update(min=min(bounds["min"], number), max=max(bounds["max"], number))
                    kind = ("serial_1x1" if geometry["grid"] == geometry["block"] == 1 else
                            "multithread" if all(geometry.values()) else "unknown_geometry")
                    execution[kind].append((start, end))
            if kernel and not intervals:
                raise ValueError("Kernel table has no completed kernel rows")
            stats = interval_metrics(intervals)
            for group in groups.values():
                group["duration_sum_s"] = group["duration_sum_ns"] / 1e9
                group["duration_share"] = group["duration_sum_ns"] / stats["duration_sum_ns"] if intervals else 0
            result = {"present": True, **stats,
                      "groups": sorted(groups.values(), key=lambda g: (-g["duration_sum_ns"], g["name"]))}
            if kernel:
                result["execution_geometry"] = {kind: {**interval_metrics(items),
                    "duration_share": sum(end-start for start, end in items) / stats["duration_sum_ns"]}
                    for kind, items in execution.items()}
            all_intervals.extend(intervals)
            return result

        kernels = summarize("CUPTI_ACTIVITY_KIND_KERNEL", True)
        copies = summarize("CUPTI_ACTIVITY_KIND_MEMCPY")
        memset = summarize("CUPTI_ACTIVITY_KIND_MEMSET")
        runtime = summarize("CUPTI_ACTIVITY_KIND_RUNTIME")
        return {"trace": str(trace), "schema": tables, "gpu_metadata": gpu_metadata,
                "kernels": kernels, "memory_copies": copies,
                "memory_sets": memset, "runtime_api": runtime,
                "recorded_activity": interval_metrics(all_intervals),
                "notes": ["All calculations use integer nanoseconds before conversion to seconds.",
                    "Kernel shares use summed kernel durations, not solver or wall time; concurrent kernels can overlap.",
                    "Interval unions merge overlaps; activity spans run from earliest start to latest end and include idle gaps.",
                    "Recorded activity includes only the listed CUPTI tables, not the entire trace capture or solver duration.",
                    "CUDA runtime API durations include host overhead and waits. They can overlap GPU work and each other; do not add them to GPU time.",
                    "serial_1x1 means gridX*gridY*gridZ == blockX*blockY*blockZ == 1; multithread does not imply efficient utilization.",
                    "Geometry/resource ranges appear only when the trace provides valid values; absent fields are unknown, not zero.",
                    "Invalid resource metadata is retained verbatim under invalid_metadata and excluded from ranges; negative values are NOT reinterpreted as unsigned or used to reject valid timing intervals."]}


def render(report):
    lines = ["Nsight Systems GPU trace analysis", report["trace"], "", "Durations below are NOT solver wall time."]
    for gpu in report["gpu_metadata"]:
        lines.append("GPU metadata: " + "; ".join(f"{key}={value}" for key, value in gpu.items()))
    for key in ("kernels", "memory_copies", "memory_sets", "runtime_api", "recorded_activity"):
        stats = report[key]
        lines.append(f"{key}: count={stats['count']} sum={stats['duration_sum_s']:.9f} s; "
                     f"union={stats['interval_union_s']:.9f} s; span={stats['activity_span_s']:.9f} s")
    lines += ["", "Kernel execution geometry:"]
    for kind, stats in report["kernels"]["execution_geometry"].items():
        lines.append(f"  {kind}: {stats['count']} launches, {stats['duration_sum_s']:.9f} s, "
                     f"{100*stats['duration_share']:.2f}% of kernel duration sum")
    for label, key in (("Top kernels", "kernels"), ("Memory copies", "memory_copies"),
                       ("CUDA runtime APIs (non-additive)", "runtime_api")):
        lines += ["", label + ":"]
        for group in report[key]["groups"][:30]:
            lines.append(f"  {group['duration_sum_s']:.9f} s | {100*group['duration_share']:.2f}% | "
                         f"{group['count']} calls | {group['name']}")
            if group["ranges"]:
                lines.append("    " + "; ".join(f"{name}={v['min']}..{v['max']}" for name, v in group["ranges"].items()))
            for field, warning in group["invalid_metadata"].items():
                lines.append(f"    WARNING: {field} unavailable for {warning['count']} launches; "
                             f"invalid raw metadata={warning['raw_values']}")
            if group["bytes_total"]:
                lines.append(f"    bytes={group['bytes_total']}")
    return "\n".join(lines + ["", "Interpretation:", *["- " + note for note in report["notes"]]]) + "\n"


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ipc_nsys_analysis_") as directory:
        trace = Path(directory) / "synthetic.sqlite"
        with sqlite3.connect(trace) as db:
            db.executescript("""
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                INSERT INTO StringIds VALUES(1,'serial kernel'),(2,'parallel kernel'),(3,'cudaSynchronize');
                CREATE TABLE TARGET_INFO_GPU(name TEXT,id INTEGER,smCount INTEGER,threadsPerWarp INTEGER);
                INSERT INTO TARGET_INFO_GPU VALUES('Synthetic GPU',0,142,32);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,
                    gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,
                    registersPerThread INTEGER,localMemoryPerThread INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                    (10,30,1,1,1,1,1,1,1,32,0),(20,50,2,2,3,1,4,2,1,48,16),(60,70,2,4,1,1,16,1,1,64,32);
                ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD localMemoryTotal INTEGER;
                UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET localMemoryTotal=-1898643456 WHERE demangledName=2;
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(5,80,3);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(0,10,1,1024),(40,60,2,2048);
            """)
        report = analyze(trace)
        assert report["kernels"]["duration_sum_ns"] == 60
        assert report["kernels"]["interval_union_ns"] == 50
        assert report["kernels"]["activity_span_ns"] == 60
        assert report["kernels"]["groups"][0]["ranges"]["launch_threads"] == {"min": 48, "max": 64}
        assert report["kernels"]["groups"][0]["invalid_metadata"]["local_bytes_total"] == {
            "count": 2, "raw_values": [{"value": -1898643456, "count": 2}]}
        assert "local_bytes_total" not in report["kernels"]["groups"][0]["ranges"]
        assert report["kernels"]["execution_geometry"]["serial_1x1"]["duration_sum_ns"] == 20
        assert report["runtime_api"]["duration_sum_ns"] == 75
        assert report["recorded_activity"]["interval_union_ns"] == 80
        assert report["gpu_metadata"][0]["smCount"] == 142
        assert not report["memory_sets"]["present"] and "NOT solver" in render(report) and "WARNING" in render(report)
        # Preserve sub-ULP differences even when timestamp origins exceed 2^53.
        with sqlite3.connect(trace) as db:
            for table in ("KERNEL", "RUNTIME", "MEMCPY"):
                db.execute("UPDATE CUPTI_ACTIVITY_KIND_" + table +
                           " SET start=start+9007199254740992,end=end+9007199254740992")
        assert analyze(trace)["kernels"]["interval_union_ns"] == 50
        with sqlite3.connect(trace) as db:
            db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=0 WHERE demangledName=1")
        try:
            analyze(trace)
        except ValueError:
            pass
        else:
            raise AssertionError("unfinished kernel was accepted")
        with sqlite3.connect(trace) as db:
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
        try:
            analyze(trace)
        except ValueError:
            pass
        else:
            raise AssertionError("empty kernel table was accepted")
        with sqlite3.connect(trace) as db:
            db.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
        try:
            analyze(trace)
        except ValueError:
            pass
        else:
            raise AssertionError("missing kernel table was accepted")
    print("Nsight SQLite analyzer self-test PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.trace is None or args.output is None:
        parser.error("--trace and --output are required")
    report = analyze(args.trace)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("gpu_trace_summary.json", "gpu_trace_summary.txt"):
        if (args.output / name).exists():
            raise FileExistsError(args.output / name)
    for name, content in (("gpu_trace_summary.json", json.dumps(report, indent=2) + "\n"),
                          ("gpu_trace_summary.txt", render(report))):
        with (args.output / name).open("x", encoding="utf-8") as output:
            output.write(content)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
